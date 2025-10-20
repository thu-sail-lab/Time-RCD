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
        self.reconstruction_head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

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
        
        mask = mask.bool()
        
        # local_embeddings: [B, seq_len, num_features, d_proj]
        reconstructed = self.reconstruction_head(local_embeddings)  # (B, seq_len, num_features, 1)
        reconstructed = reconstructed.view(batch_size, seq_len, num_features)  
        
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, num_features)  # (B, seq_len, num_features)
        reconstruction_loss = F.mse_loss(
            reconstructed[mask_expanded],
            original_time_series[mask_expanded]
        )
        return reconstruction_loss
    
def create_random_mask(time_series: torch.Tensor,  #(B, max_seq_len, num_features)
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
    # concatenated = torch.cat(time_series_tensors, dim=0)  # (total_length, num_features)
    # mean = concatenated.mean(dim=0, keepdim=True)  # (1, num_features)
    # std = concatenated.std(dim=0, keepdim=True)  # (1, num_features)
    # std = std + 1e-4
    # time_series_tensors_std = [(ts - mean) / std for ts in time_series_tensors]
    # normal_time_series_tensors_std = [(nts - mean) / std for nts in normal_time_series_tensors]
    # time_series_tensors = time_series_tensors_std
    # normal_time_series_tensors = normal_time_series_tensors_std

    means = []
    stds = []
    for i in range(len(time_series_tensors)):
        ts = time_series_tensors[i]
        mean = ts.mean(dim=0, keepdim=True)
        std = ts.std(dim=0, keepdim=True) + 1e-4
        means.append(mean)
        stds.append(std)
        time_series_tensors[i] = (ts - mean) / std
    for i in range(len(normal_time_series_tensors)):
        nts = normal_time_series_tensors[i]
        mean = means[i]
        std = stds[i]
        normal_time_series_tensors[i] = (nts - mean) / std

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


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False