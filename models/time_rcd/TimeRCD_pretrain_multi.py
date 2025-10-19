import datetime
import itertools
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import random
import numpy as np
from typing import Tuple, List, Dict, Any, Union, Optional
from dataclasses import dataclass

from .dataset import ChatTSTimeRCDPretrainDataset
from .ts_encoder_bi_bias import TimeSeriesEncoder
from .time_rcd_config import TimeRCDConfig, default_config

import warnings

warnings.filterwarnings("ignore")


# PYTHONPATH=/home2/lijinbo/Projects/Time_RCD-master/ python src/models/Moirai/TimeRCD_pretrain_multi.py
@dataclass
class PretrainBatch:
    """Batch structure for pretraining tasks."""
    time_series: torch.Tensor
    labels: torch.Tensor
    masked_time_series: torch.Tensor
    mask_indices: torch.Tensor


class TimeSeriesPretrainModel(nn.Module):
    """Model for time series pretraining with masked reconstruction and anomaly detection."""

    def __init__(self, config: TimeRCDConfig):
        super().__init__()
        self.config = config

        # Extract TimeSeriesEncoder parameters from config
        ts_config = config.ts_config
        self.ts_encoder = TimeSeriesEncoder(
            d_model=ts_config.d_model,
            d_proj=ts_config.d_proj,
            patch_size=ts_config.patch_size,
            num_layers=ts_config.num_layers,
            num_heads=ts_config.num_heads,
            d_ff_dropout=ts_config.d_ff_dropout,
            use_rope=ts_config.use_rope,
            num_features=ts_config.num_features,
            activation=ts_config.activation
        )

        # Masked reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.ts_config.d_proj, config.ts_config.d_proj * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ts_config.d_proj * 4, config.ts_config.d_proj * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ts_config.d_proj * 4, 1)  # (B, seq_len, num_features, 1)
        )

        # Anomaly detection head
        self.anomaly_head = nn.Sequential(
            nn.Linear(config.ts_config.d_proj, config.ts_config.d_proj // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ts_config.d_proj // 2, 2)  # (B, seq_len, num_features, 2) for binary classification
        )

    def forward(self, time_series: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Forward pass through the encoder."""
        local_embeddings = self.ts_encoder(time_series, mask)
        return local_embeddings

    def masked_reconstruction_loss(self,
                                   local_embeddings: torch.Tensor,  # (B, seq_len, num_features, d_proj)
                                   original_time_series: torch.Tensor,  # (B, seq_len, num_features),
                                   mask: torch.Tensor  # (B, seq_len)
                                   ) -> torch.Tensor:
        """Compute masked reconstruction loss."""
        batch_size, seq_len, num_features = original_time_series.shape
        patch_size = self.config.ts_config.patch_size

        # 确保数据类型一致
        mask = mask.bool()

        # 只对 masked 的位置计算损失
        # local_embeddings: [B, seq_len, num_features, d_proj]
        # 通过重构头预测原始值
        reconstructed = self.reconstruction_head(local_embeddings)  # (B, seq_len, num_features, 1)
        reconstructed = reconstructed.view(batch_size, seq_len, num_features)

        # 只对被 mask 的位置计算损失
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_features)  # (B, seq_len, num_features)
        reconstruction_loss = F.mse_loss(
            reconstructed[mask_expanded],
            original_time_series[mask_expanded]
        )
        return reconstruction_loss

    def anomaly_detection_loss(self,
                               local_embeddings: torch.Tensor,  # (B, seq_len, num_features, d_proj)
                               labels: torch.Tensor) -> torch.Tensor:  # (B, seq_len)
        """Compute anomaly detection loss for each timestep."""
        # Project local embeddings to anomaly scores
        logits = self.anomaly_head(local_embeddings)  # (B, seq_len, num_features, 2)
        logits = torch.mean(logits, dim=-2)  # Average over num_features to get (B, seq_len, 2)

        # Reshape for loss computation
        batch_size, seq_len, _ = logits.shape
        logits = logits.view(-1, 2)  # (B*seq_len, 2)
        labels = labels.view(-1)  # (B*seq_len)
        labels = (labels > 0.5).long()
        # Create mask for valid labels (not padding)
        valid_mask = (labels != -1)

        # Compute loss only on valid timesteps
        if valid_mask.sum() > 0:
            anomaly_loss = F.cross_entropy(
                logits[valid_mask],
                labels[valid_mask]
            )
        else:
            anomaly_loss = torch.tensor(0.0, device=logits.device)

        return anomaly_loss


def create_random_mask(time_series: torch.Tensor,  # (B, max_seq_len, num_features)
                       attention_mask: torch.Tensor,  # (B, max_seq_len)
                       mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create random mask for time series patches, only masking valid sequence parts."""
    batch_size, seq_len, num_features = time_series.shape
    patch_size = default_config.ts_config.patch_size

    mask = torch.zeros(batch_size, seq_len)  # (B, max_seq_len)

    for i in range(batch_size):
        # Get valid sequence length for this sample
        valid_length = attention_mask[i].sum().item()

        # Calculate number of patches in valid sequence
        num_valid_patches = (valid_length - 1) // patch_size + 1
        num_masked = int(num_valid_patches * mask_ratio)

        if num_masked > 0:
            # Only select patches from valid sequence
            masked_patches = torch.randperm(num_valid_patches)[:num_masked]
            for j in masked_patches:
                start_idx = j * patch_size
                end_idx = min((j + 1) * patch_size, valid_length)  # Don't exceed valid length
                mask[i, start_idx:end_idx] = 1

    # Create masked time series - only mask valid parts
    masked_time_series = time_series.clone()
    mask_indices = mask.bool() & attention_mask  # Only mask where both mask and attention_mask are True
    mask_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, num_features)  # (B, max_seq_len, num_features)
    masked_time_series[mask_expanded] = torch.randn_like(masked_time_series[mask_expanded]) * 0.1

    # Update mask to only include valid parts
    mask = mask * attention_mask.float()

    return masked_time_series, mask  # (B, max_seq_len, num_features), (B, max_seq_len)


def collate_fn(batch):
    """Collate function for pretraining dataset."""
    time_series_list, normal_time_series_list, labels_list, attribute_list = zip(*batch)

    # Convert to tensors and pad sequences
    if time_series_list[0].ndim == 1:
        time_series_tensors = [ts.unsqueeze(-1) for ts in time_series_list]  # Add feature dimension
        normal_time_series_tensors = [nts.unsqueeze(-1) for nts in normal_time_series_list]
    else:
        time_series_tensors = [ts for ts in time_series_list]
        normal_time_series_tensors = [nts for nts in normal_time_series_list]

    # standardize time series
    concatenated = torch.cat(time_series_tensors, dim=0)  # (total_length, num_features)
    mean = concatenated.mean(dim=0, keepdim=True)  # (1, num_features)
    std = concatenated.std(dim=0, keepdim=True)  # (1, num_features)
    std = std + 1e-4
    time_series_tensors_std = [(ts - mean) / std for ts in time_series_tensors]
    normal_time_series_tensors_std = [(nts - mean) / std for nts in normal_time_series_tensors]
    time_series_tensors = time_series_tensors_std
    normal_time_series_tensors = normal_time_series_tensors_std

    # labels_tensor = torch.stack(labels_list)
    labels = [label for label in labels_list]
    # Pad time series to same length
    padded_time_series = torch.nn.utils.rnn.pad_sequence(
        time_series_tensors, batch_first=True, padding_value=0.0
    )  # (B, max_seq_len, num_features)
    padded_normal_time_series = torch.nn.utils.rnn.pad_sequence(
        normal_time_series_tensors, batch_first=True, padding_value=0.0
    )  # (B, max_seq_len, num_features)
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-1
    )  # (B, max_seq_len)

    sequence_lengths = [ts.size(0) for ts in time_series_tensors]
    B, max_seq_len, num_features = padded_time_series.shape
    attention_mask = torch.zeros(B, max_seq_len, dtype=torch.bool)  # (B, max_seq_len)
    for i, length in enumerate(sequence_lengths):
        attention_mask[i, :length] = True

        # Create random masks for reconstruction task - only mask valid sequence parts
    masked_time_series, mask = create_random_mask(padded_time_series, attention_mask)

    return {
        'time_series': padded_time_series,
        'normal_time_series': padded_normal_time_series,
        'masked_time_series': masked_time_series,
        'mask': mask,  # for reconstruction task
        'labels': padded_labels,
        'attention_mask': attention_mask,  # for padding
        'attribute': attribute_list
    }

def test_collate_fn(batch):
    """Collate function for pretraining dataset."""
    # Unpack the batch correctly - batch is a list of (time_series, mask) tuples
    time_series_list, mask_list = zip(*batch)

    # Convert to tensors and pad sequences
    # if time_series_list[0].ndim == 1:
    #     time_series_tensors = [ts.unsqueeze(-1) for ts in time_series_list]  # Add feature dimension
    # else:
    #     time_series_tensors = [ts for ts in time_series_list]

    # Stack into batch format instead of concatenating
    # This maintains the batch dimension: (B, seq_len, num_features)
    batched_time_series = torch.stack(time_series_list, dim=0)
    print(f"batched_time_series shape: {batched_time_series.shape}")
    # Stack masks into batch format: (B, seq_len)
    batched_mask = torch.stack(mask_list, dim=0)
    print(f"batched_mask shape: {batched_mask.shape}")

    return {
        'time_series': batched_time_series,
        'attention_mask': batched_mask,  # for padding
    }

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_distributed(rank: int, world_size: int, config: TimeRCDConfig) -> None:
    """Setup distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.dist_port

    try:
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=30)
        )
        torch.cuda.set_device(rank)

        if rank == 0:
            print(f"Successfully initialized distributed training on rank {rank} with world size {world_size}")

    except Exception as e:
        print(f"Rank {rank}: Initialization failed with error: {e}")
        raise e


def cleanup_distributed() -> None:
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def evaluate_epoch(test_loader: DataLoader,
                   model: nn.Module,
                   config: TimeRCDConfig,
                   device: torch.device,
                   rank: int) -> float:
    """Evaluate model on test dataset."""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_anomaly_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in itertools.islice(test_loader, min(len(test_loader), config.test_batch_limit)):
            # Move data to device
            time_series = batch['time_series'].to(device)
            masked_time_series = batch['masked_time_series'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            local_embeddings = model(masked_time_series, attention_mask & (~mask.bool()))

            # Compute losses
            recon_loss = model.module.masked_reconstruction_loss(
                local_embeddings, time_series, mask
            )
            anomaly_loss = model.module.anomaly_detection_loss(local_embeddings, labels)

            total_loss_batch = recon_loss + anomaly_loss
            total_loss += total_loss_batch.item()
            total_recon_loss += recon_loss.item()
            total_anomaly_loss += anomaly_loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0
    avg_anomaly_loss = total_anomaly_loss / num_batches if num_batches > 0 else 0.0

    if rank == 0:
        print(f"Validation Results:")
        print(f"  Average Total Loss: {avg_loss:.4f}")
        print(f"  Average Recon Loss: {avg_recon_loss:.4f}")
        print(f"  Average Anomaly Loss: {avg_anomaly_loss:.4f}")

    return avg_loss


def train_epoch(train_loader: DataLoader,
                model: nn.Module,
                optimizer: optim.Optimizer,
                config: TimeRCDConfig,
                device: torch.device,
                epoch: int,
                rank: int,
                scaler: Optional[torch.cuda.amp.GradScaler] = None) -> float:
    """Train for one epoch with multiple pretraining tasks."""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_anomaly_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

        optimizer.zero_grad()

        # Move data to device
        time_series = batch['time_series'].to(device)  # (B, max_seq_len, num_features)
        masked_time_series = batch['masked_time_series'].to(device)
        mask = batch['mask'].to(device)  # (B, max_seq_len)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        if config.mixed_precision and scaler is not None:
            with torch.amp.autocast('cuda'):
                local_embeddings = model(masked_time_series, attention_mask & (~mask.bool()))

                recon_loss = model.module.masked_reconstruction_loss(
                    local_embeddings, time_series, mask
                )
                anomaly_loss = model.module.anomaly_detection_loss(local_embeddings, labels)

            total_loss_batch = recon_loss + anomaly_loss
            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            local_embeddings = model(masked_time_series, attention_mask & (~mask.bool()))

            recon_loss = model.module.masked_reconstruction_loss(
                local_embeddings, time_series, mask
            )
            anomaly_loss = model.module.anomaly_detection_loss(local_embeddings, labels)

            total_loss_batch = recon_loss + anomaly_loss
            total_loss_batch.backward()
            optimizer.step()

        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_recon_loss += recon_loss.item()
        total_anomaly_loss += anomaly_loss.item()
        num_batches += 1

        # Log progress based on log_freq
        if rank == 0 and batch_idx % config.log_freq == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}")
            print(f"  Total Loss: {total_loss_batch.item():.4f}")
            print(f"  Recon Loss: {recon_loss.item():.4f}")
            print(f"  Anomaly Loss: {anomaly_loss.item():.4f}")

    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_anomaly_loss = total_anomaly_loss / num_batches

    if rank == 0:
        print(f"Epoch {epoch} completed:")
        print(f"  Average Total Loss: {avg_loss:.4f}")
        print(f"  Average Recon Loss: {avg_recon_loss:.4f}")
        print(f"  Average Anomaly Loss: {avg_anomaly_loss:.4f}")

    return avg_loss


def save_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer,
                    config: TimeRCDConfig,
                    epoch: int,
                    avg_loss: float,
                    rank: int = 0,
                    is_best: bool = False) -> None:
    """Save model checkpoint."""
    if rank != 0:
        return

    # Extract model state dict (handle DDP wrapper)
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config.to_dict()
    }

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Always save the latest checkpoint
    latest_path = os.path.join(config.checkpoint_dir, "pretrain_checkpoint_latest.pth")
    torch.save(checkpoint, latest_path)

    # Save the checkpoint at specified frequency
    if epoch % config.save_freq == 0 or epoch == config.num_epochs - 1:
        save_path = os.path.join(config.checkpoint_dir, f"pretrain_checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path} (epoch {epoch}, loss: {avg_loss:.4f})")

    # Save best model if this is the best validation loss
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, "pretrain_checkpoint_best.pth")
        torch.save(checkpoint, best_path)
        print(f"New best model saved to {best_path} (epoch {epoch}, val_loss: {avg_loss:.4f})")

        # Save just the time series encoder for downstream tasks
        if hasattr(model, 'module'):
            ts_encoder_state = model.module.ts_encoder.state_dict()
        else:
            ts_encoder_state = model.ts_encoder.state_dict()

        best_encoder_path = os.path.join(config.checkpoint_dir, "pretrained_ts_encoder.pth")
        torch.save(ts_encoder_state, best_encoder_path)
        print(f"Best pretrained time series encoder saved to {best_encoder_path}")


def train_multiple_datasets(dataset_filenames: List[str], config: TimeRCDConfig) -> None:
    """Train on multiple datasets sequentially with model weight continuation."""
    print(f'\n{"=" * 50}')
    print(f"Starting Multi-Dataset Sequential Training")
    print(f"Number of datasets: {len(dataset_filenames)}")
    print(f"Datasets: {dataset_filenames}")
    print("Training Mode: Continuous (model weights carried over between datasets)")
    print("=" * 50)

    # Parse GPU IDs from config
    gpu_ids = [int(x.strip()) for x in config.cuda_devices.split(',')]
    world_size = len(gpu_ids)

    # Set CUDA_VISIBLE_DEVICES
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_devices

    # Global checkpoint path for model continuation
    global_checkpoint_path = None
    # global_checkpoint_path = "experiments/checkpoints/pretrain_single_16ps/pretrain_checkpoint_latest.pth"

    for dataset_idx, filename in enumerate(dataset_filenames):
        print(f"\n{'=' * 50}")
        print(f"Training on Dataset {dataset_idx + 1}/{len(dataset_filenames)}: {filename}")
        if global_checkpoint_path is not None:
            print(f"Continuing from previous dataset's trained model...")
        print(f"{'=' * 50}")

        batch_size_list = [256, 64, 64, 32, 32, 16, 16, 48,
                           16, 16, 16, 32, 16, 16, 16, 16,
                           16, 16, 16, 16, 12, 12, 12, 16,
                           12, 12, 12, 12, 12, 12, 12, 16,
                           12, 12, 12, 12, 12, 12, 12, 12,
                           12, 12, 12, 12, 12, 12, 12, 12,
                           12, 12, 12, 12, 12, 12, 12, 8]
        num_features = int(os.path.splitext(filename)[0].split('_')[-1])
        print(
            f"Using batch size: {batch_size_list[num_features - 1] if num_features <= len(batch_size_list) else batch_size_list[-1]} for {filename}")
        if num_features <= len(batch_size_list):
            batch_size = batch_size_list[num_features - 1]
        else:
            batch_size = batch_size_list[-1]
        config.batch_size = batch_size

        # Create dataset-specific checkpoint directory
        original_checkpoint_dir = config.checkpoint_dir
        config.checkpoint_dir = os.path.join(original_checkpoint_dir, f"{filename}")
        os.makedirs(config.checkpoint_dir, exist_ok=True)

        # Set the checkpoint path for model continuation
        config.continuation_checkpoint = global_checkpoint_path

        config.ts_config.num_features = num_features
        if world_size == 1:
            # Single GPU training
            print(f"Running single GPU pretraining for {filename}...")
            train_worker(0, 1, config, filename)
        else:
            # Multi-GPU distributed training
            print(f"Running distributed pretraining for {filename}...")
            mp.spawn(
                train_worker,
                args=(world_size, config, filename),
                nprocs=world_size,
                join=True
            )

        # Update global checkpoint path for next dataset
        global_checkpoint_path = os.path.join(config.checkpoint_dir, "pretrain_checkpoint_best.pth")

        # Restore original checkpoint directory
        config.checkpoint_dir = original_checkpoint_dir

        print(f"Completed training on dataset: {filename}")
        if dataset_idx < len(dataset_filenames) - 1:
            print(f"Model weights will be loaded for next dataset training...")

    print(f"\n{'=' * 50}")
    print("Multi-Dataset Sequential Training Completed!")
    print(f"All {len(dataset_filenames)} datasets have been processed with model continuation.")
    print(f"{'=' * 50}")


def train_worker(rank: int, world_size: int, config: TimeRCDConfig, filename: str = None) -> None:
    """Training worker function for each process."""
    print(f"Running DDP on rank {rank} with world_size {world_size} for dataset: {filename}")

    # Setup distributed training
    setup_distributed(rank, world_size, config)

    # Set device for this process
    device = torch.device(f"cuda:{rank}")

    # Set random seed
    set_seed(config.seed + rank)

    try:
        # Initialize model
        model = TimeSeriesPretrainModel(config).to(device)

        # Load checkpoint if continuing from previous dataset
        checkpoint = None
        if hasattr(config, 'continuation_checkpoint') and config.continuation_checkpoint and os.path.exists(
                config.continuation_checkpoint):
            if rank == 0:
                print(f"Loading checkpoint from previous dataset: {config.continuation_checkpoint}")
            checkpoint = torch.load(config.continuation_checkpoint, map_location=device)

            # Handle DDP state_dict compatibility
            state_dict = checkpoint['model_state_dict']

            # Remove 'module.' prefix if it exists (from DDP wrapped model)
            if any(key.startswith('module.') for key in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('module.'):
                        new_key = key[7:]  # Remove 'module.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict

            model.load_state_dict(state_dict, strict=False)
            if rank == 0:
                print(f"Successfully loaded model weights from previous dataset training")

        # Wrap model with DDP
        # model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        model = DDP(model, device_ids=[rank])

        # Setup optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Load optimizer state if continuing and optimizer state exists
        if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if rank == 0:
                    print("Successfully loaded optimizer state from previous dataset training")
            except Exception as e:
                if rank == 0:
                    print(f"Warning: Could not load optimizer state: {e}")
                    print("Continuing with fresh optimizer state")
                    print("This is normal when model architecture or optimizer parameters change")

        # Setup mixed precision scaler
        scaler = torch.amp.GradScaler() if config.mixed_precision else None

        # Load data
        train_dataset = ChatTSTimeRCDPretrainDataset(config.pretrain_data_path, filename, split="train")
        test_dataset = ChatTSTimeRCDPretrainDataset(config.pretrain_data_path, filename, split="test")

        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )

        # Create test sampler and loader for validation
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True
        )

        # Early stopping parameters
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = getattr(config, 'early_stopping_patience', 10)

        # Training loop
        if rank == 0:
            dataset_name = filename if filename else "default"
            continuation_info = ""
            if hasattr(config, 'continuation_checkpoint') and config.continuation_checkpoint and os.path.exists(
                    config.continuation_checkpoint):
                continuation_info = " (continuing from previous dataset)"
            print(
                f"Starting pretraining for {config.num_epochs} epochs on dataset {dataset_name}{continuation_info}...")
            print(f"Total training batches per process: {len(train_loader)}")
            print(f"Total validation batches per process: {min(config.test_batch_limit, len(test_loader))}")
            print(f"Early stopping patience: {early_stopping_patience} epochs")
            print(f"Tasks: Masked Reconstruction + Anomaly Detection")

        for epoch in range(config.num_epochs):
            # Set epoch for distributed samplers
            train_sampler.set_epoch(epoch)
            test_sampler.set_epoch(epoch)

            # Training phase
            avg_train_loss = train_epoch(train_loader, model, optimizer,
                                         config, device, epoch, rank, scaler)

            # Validation phase
            avg_val_loss = evaluate_epoch(test_loader, model, config, device, rank)

            # Check if this is the best model so far
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                patience_counter = 0
                if rank == 0:
                    print(f"New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if rank == 0:
                    print(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")

            # Save checkpoint with best model flag
            save_checkpoint(model, optimizer, config, epoch, avg_val_loss, rank, is_best)

            # Early stopping check
            if patience_counter >= early_stopping_patience:
                if rank == 0:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                break


    finally:
        # Clean up distributed training
        cleanup_distributed()


def main() -> None:
    # PYTHONPATH=/home2/lijinbo/Projects/Time_RCD-master/ python src/models/Moirai/TimeRCD_pretrain_multi.py
    """Main function to launch distributed pretraining."""
    # Load configuration
    config = default_config

    # Update config for pretraining
    config.num_epochs = 50
    config.learning_rate = 5e-4  # Higher learning rate for pretraining
    config.batch_size = 64
    config.ts_config.patch_size = 16
    config.checkpoint_dir = "experiments/checkpoints/pretrain_single_16ps"
    config.cuda_devices = "1"
    config.mixed_precision = False
    config.dist_port = "12198"
    config.early_stopping_patience = 7  # Stop training if validation loss doesn't improve for 10 epochs
    config.pretrain_data_path = "/home2/lijinbo/Projects/dataset/ChatTS/single_samples_activate_100_10000"

    # ===== Multidataset Training Configuration =====
    # Change to True for multi-dataset training
    use_multi_dataset_training = False
    config.continuation_checkpoint = "experiments/checkpoints/pretrain_single_16ps/pretrain_checkpoint_latest.pth"
    # Filename for single dataset training
    single_dataset_filename = "dataset_0_1.pkl"
    # Filename list for multi-dataset training
    # dataset_filenames = [
    #     # "dataset_0_1.pkl",
    #     # "dataset_1_1.pkl",
    #     # "dataset_2_1.pkl",
    #     # "dataset_7_8.pkl",
    #     # "dataset_8_12.pkl",
    #     # "dataset_9_16.pkl",
    #     "dataset_10_20.pkl",
    # ]

    dataset_filenames = [
        # "dataset_0_1.pkl",
        # "dataset_1_1.pkl",
        # "dataset_2_1.pkl",
        # "dataset_3_2.pkl",
        # "dataset_5_4.pkl",
        # "dataset_4_3.pkl",
        "dataset_6_5.pkl",
    ]

    # dataset_filenames = [
    #     # "dataset_0_1.pkl",
    #     # "dataset_1_1.pkl",
    #     # "dataset_2_1.pkl",
    #     # "dataset_10_24.pkl",
    #     "dataset_11_32.pkl",
    #     "dataset_12_40.pkl",
    #     "dataset_13_48.pkl",
    #     "dataset_14_56.pkl"
    # ]

    # dataset_filenames = [
    #     "dataset_0_1.pkl",
    #     "dataset_1_2.pkl",
    #     "dataset_2_3.pkl",
    #     "dataset_3_4.pkl",
    #     "dataset_4_5.pkl",
    # ]

    # Parse GPU IDs from config
    gpu_ids = [int(x.strip()) for x in config.cuda_devices.split(',')]
    world_size = len(gpu_ids)

    print(f"Using GPUs: {gpu_ids}")
    print(f"World size: {world_size}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', config.cuda_devices)}")
    print("=" * 50)
    print("AnomalyLLava Time Series Pretraining")
    print("Tasks:")
    print("  1. Masked Reconstruction - Reconstruct masked time series patches")
    print("  2. Anomaly Detection - Binary classification of normal/anomalous series")
    print("Features:")
    print("  - Early stopping with validation loss monitoring")
    print("  - Best model checkpoint saving")
    print(f"  - Early stopping patience: {config.early_stopping_patience} epochs")
    if use_multi_dataset_training:
        print("  - Sequential multi-dataset training with model weight continuation")
    print("=" * 50)

    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    if use_multi_dataset_training:
        # Multi-dataset training
        print(f"Training Mode: Multi-Dataset Sequential ({len(dataset_filenames)} datasets)")
        print(f"Datasets will be trained sequentially with model weight continuation")
        train_multiple_datasets(dataset_filenames, config)
    else:
        # Single dataset training
        print(f"Training Mode: Single Dataset ({single_dataset_filename})")
        # Set CUDA_VISIBLE_DEVICES
        os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_devices

        num_features = int(os.path.splitext(single_dataset_filename)[0].split('_')[-1])
        config.ts_config.num_features = num_features
        if world_size == 1:
            # Single GPU training
            print("Running single GPU pretraining...")
            train_worker(0, 1, config, single_dataset_filename)
        else:
            # Multi-GPU distributed training
            print("Running distributed pretraining...")
            mp.spawn(
                train_worker,
                args=(world_size, config, single_dataset_filename),
                nprocs=world_size,
                join=True
            )


if __name__ == "__main__":
    main()