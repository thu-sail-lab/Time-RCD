import tqdm
import os
import textwrap
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
import warnings
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from .time_rcd.dataset import ChatTSTimeRCDPretrainDataset
from .time_rcd.TimeRCD_pretrain_multi import TimeSeriesPretrainModel, create_random_mask, collate_fn, test_collate_fn
from .time_rcd.time_rcd_config import TimeRCDConfig, default_config
from utils.dataset import TimeRCDDataset

class TimeRCDPretrainTester:
    """Tester class for visualizing pretrained model results."""

    def __init__(self, checkpoint_path: str, config: TimeRCDConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.win_size = config.win_size
        self.batch_size = config.batch_size
        # Load model
        self.model = TimeSeriesPretrainModel(config).to(self.device)
        self.load_checkpoint(checkpoint_path)
        self.model.eval()

        print(f"Model loaded on device: {self.device}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DDP training)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # Remove 'module.' prefix
            else:
                new_key = key
            new_state_dict[new_key] = value

        self.model.load_state_dict(new_state_dict)
        print(f"Successfully loaded checkpoint from {checkpoint_path}")

    def predict(self, batch):
        """Run inference on a batch."""
        with torch.no_grad():
            # Move data to device
            time_series = batch['time_series'].to(self.device)
            normal_time_series = batch['normal_time_series'].to(self.device)
            masked_time_series = batch['masked_time_series'].to(self.device)
            attribute = batch['attribute']
            batch_size, seq_len, num_features = time_series.shape

            # 对时间序列标准化
            time_series = (time_series - time_series.mean(dim=1, keepdim=True)) / (time_series.std(dim=1, keepdim=True) + 1e-8)
            masked_time_series = (masked_time_series - masked_time_series.mean(dim=1, keepdim=True)) / (masked_time_series.std(dim=1, keepdim=True) + 1e-8)

            mask = batch['mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Get embeddings
            local_embeddings = self.model(
                time_series=time_series,
                mask=attention_mask)

            # Get reconstruction
            reconstructed = self.model.reconstruction_head(local_embeddings)
            reconstructed = reconstructed.view(batch_size, seq_len, num_features)  # (B, seq_len, num_features)

            # Get anomaly predictions
            anomaly_logits = self.model.anomaly_head(local_embeddings)
            anomaly_logits = torch.mean(anomaly_logits, dim=-2)  # (B, seq_len, 2)
            anomaly_probs = F.softmax(anomaly_logits, dim=-1)[..., 1]  # Probability of anomaly (B, seq_len)

            return {
                'original': time_series.cpu(),
                'normal': normal_time_series.cpu(),
                'masked': masked_time_series.cpu(),
                'reconstructed': reconstructed.cpu(),
                'mask': mask.cpu(),
                'anomaly_probs': anomaly_probs.cpu(),
                'true_labels': labels.cpu(),
                'attention_mask': attention_mask.cpu(),
                'attribute': attribute
            }

    def visualize_single_sample(self, results, sample_idx=0, save_path=None):
        """Visualize results for a single time series sample."""
        # Extract data for the specified sample
        original = results['original'][sample_idx].squeeze(-1).numpy()  # (seq_len, num_features) / (seq_len,)
        normal = results['normal'][sample_idx].squeeze(-1).numpy()
        masked = results['masked'][sample_idx].squeeze(-1).numpy()
        reconstructed = results['reconstructed'][sample_idx].squeeze(-1).numpy()
        mask = results['mask'][sample_idx].numpy().astype(bool)
        anomaly_probs = results['anomaly_probs'][sample_idx].numpy()  # (seq_len,)
        true_labels = results['true_labels'][sample_idx].numpy()  # (seq_len,)
        attention_mask = results['attention_mask'][sample_idx].numpy().astype(bool)
        attribute = results['attribute'][sample_idx]

        # Only consider valid sequence length
        valid_length = attention_mask.sum()
        original = original[:valid_length]
        normal = normal[:valid_length]
        masked = masked[:valid_length]
        reconstructed = reconstructed[:valid_length]
        mask = mask[:valid_length]
        anomaly_probs = anomaly_probs[:valid_length]
        true_labels = true_labels[:valid_length]

        # Create time axis
        time_axis = np.arange(len(original))

        assert original.ndim == normal.ndim == reconstructed.ndim == masked.ndim, "Original, normal, reconstructed, and masked time series must have the same dimensions."
        if original.ndim == 1:
            # Create subplots
            fig, axes = plt.subplots(3, 1, figsize=(15, 12))

            # 1. Reconstruction visualization
            ax1 = axes[0]
            ax1.plot(time_axis, original, 'b-', label='Original', linewidth=2, alpha=0.8)
            ax1.plot(time_axis, masked, 'g--', label='Masked Input', linewidth=1.5, alpha=0.7)
            ax1.plot(time_axis[mask], reconstructed[mask], 'ro',
                     label='Reconstructed', markersize=4, alpha=0.8)

            # Highlight masked regions
            mask_regions = []
            in_mask = False
            start_idx = 0

            for i, is_masked in enumerate(mask):
                if is_masked and not in_mask:
                    start_idx = i
                    in_mask = True
                elif not is_masked and in_mask:
                    mask_regions.append((start_idx, i - 1))
                    in_mask = False

            if in_mask:  # Handle case where mask continues to the end
                mask_regions.append((start_idx, len(mask) - 1))

            for start, end in mask_regions:
                ax1.axvspan(start, end, alpha=0.2, color='red',
                            label='Masked Region' if start == mask_regions[0][0] else "")

            ax1.set_title('Time Series Reconstruction', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Anomaly detection visualization
            ax2 = axes[1]
            ax2.plot(time_axis, normal, 'g-', label='Normal Time Series', linewidth=1, alpha=0.6)
            ax2.plot(time_axis, original, 'b-', label='Anomalous Time Series', linewidth=1, alpha=0.6)

            # Color background based on true anomaly labels
            anomaly_regions = []
            in_anomaly = False
            start_idx = 0

            for i, is_anomaly in enumerate(true_labels > 0.5):
                if is_anomaly and not in_anomaly:
                    start_idx = i
                    in_anomaly = True
                elif not is_anomaly and in_anomaly:
                    anomaly_regions.append((start_idx, i - 1))
                    in_anomaly = False

            if in_anomaly:
                anomaly_regions.append((start_idx, len(true_labels) - 1))

            for start, end in anomaly_regions:
                ax2.axvspan(start, end, alpha=0.3, color='red',
                            label='True Anomaly' if start == anomaly_regions[0][0] else "")

            # Plot predicted anomaly probabilities
            ax2_twin = ax2.twinx()
            ax2_twin.plot(time_axis, anomaly_probs, 'r-', label='Anomaly Probability',
                          linewidth=2, alpha=0.8)
            ax2_twin.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7,
                             label='Threshold (0.5)')
            ax2_twin.set_ylabel('Anomaly Probability', color='red')
            ax2_twin.set_ylim(0, 1)

            ax2.set_title('Anomaly Detection Results', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Time Series Value', color='blue')

            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            ax2.grid(True, alpha=0.3)

            # 3. Performance metrics visualization
            ax3 = axes[2]

            # Calculate reconstruction error for masked regions
            if mask.sum() > 0:
                recon_error = np.abs(original[mask] - reconstructed[mask])
                ax3.bar(np.arange(len(recon_error)), recon_error,
                        alpha=0.7, color='orange', label='Reconstruction Error')
                ax3.set_title('Reconstruction Error (Masked Regions Only)',
                              fontsize=14, fontweight='bold')
                ax3.set_xlabel('Masked Time Step Index')
                ax3.set_ylabel('Absolute Error')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No masked regions in this sample',
                         ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                ax3.set_title('Reconstruction Error', fontsize=14, fontweight='bold')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')

            plt.show()

        elif original.ndim == 2:
            _, num_features = original.shape

            fig_height = 4 * num_features + 2
            fig, axes = plt.subplots(num_features, 1, figsize=(16, fig_height))
            plt.subplots_adjust(top=0.85, hspace=0.2, left=0.08, right=0.92, bottom=0.08)

            anomaly_regions = []
            in_anomaly = False
            start_idx = 0
            for i, is_anomaly in enumerate(true_labels > 0.5):
                if is_anomaly and not in_anomaly:
                    start_idx = i
                    in_anomaly = True
                elif not is_anomaly and in_anomaly:
                    anomaly_regions.append((start_idx, i - 1))
                    in_anomaly = False
            if in_anomaly:
                anomaly_regions.append((start_idx, len(true_labels) - 1))

            for feature_idx in range(num_features):
                ax = axes[feature_idx]
                ax.plot(time_axis, original[:, feature_idx], 'b-',
                        linewidth=1, label=f'Anomalous Time Series', alpha=0.8)
                ax.plot(time_axis, normal[:, feature_idx], 'g-',
                        linewidth=1, label='Normal Time Series', alpha=0.8)
                y_min, y_max = ax.get_ylim()
                shift = y_max - y_min
                ax.set_ylim(y_min - shift, y_max)

                for start, end in anomaly_regions:
                    if start == end:
                        ax.axvspan(start - 0.5, start + 0.5, alpha=0.3, color='grey',
                                   label='True Anomaly Region' if start == anomaly_regions[0][
                                       0] and feature_idx == 0 else "")
                    else:
                        ax.axvspan(start, end, alpha=0.3, color='grey',
                                   label='True Anomaly Region' if start == anomaly_regions[0][
                                       0] and feature_idx == 0 else "")

                ax2 = ax.twinx()
                ax2.plot(time_axis, anomaly_probs, 'r-', linewidth=1,
                         label='Anomaly Score', alpha=0.9)
                ax2.set_ylim(0, 1.5)
                ax2.set_ylabel('Anomaly Score', fontsize=12)
                ax.set_ylabel(f'Value', fontsize=12)
                if feature_idx == num_features - 1:
                    ax.set_xlabel('Time Steps', fontsize=12)
                else:
                    ax.set_xticklabels([])

                ax.set_title(f'Feature {feature_idx} - Time Series & Anomaly Score',
                             fontsize=16, pad=10)
                ax.grid(True, alpha=0.3)

                if feature_idx == 0:
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2,
                              loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=14)

            anomalies = []
            isendo = attribute['is_endogenous']
            edges = attribute['dag']
            for idx, item in enumerate(attribute['attribute_list']):
                for k, v in item['anomalies'].items():
                    anomalies.append((f"feature_{idx}_{k[2:]}", v))
            anomalies_str = ', '.join([f"{k}: {v}" for k, v in anomalies])
            wrap_width = 100
            wrapped_anomalies = textwrap.fill(f"Anomalies: {anomalies_str}", width=wrap_width)
            wrapped_edges = textwrap.fill(f"Edges: {str(edges)}", width=wrap_width)
            title = f"Multivariate Time Series Visualization\n{isendo}_{wrapped_anomalies}\n{wrapped_edges}"
            fig.suptitle(title, fontsize=22, y=0.95, ha='center', va='top')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()


        else:
            raise ValueError("Unsupported original data shape: {}".format(original.shape))

    def test_model(self, data_path: str, filename: str, num_samples: int = 5, save_dir: str = None,
                   max_test_data: int = 100):
        """Test the model on a dataset and visualize results."""
        # Load test dataset
        full_test_dataset = ChatTSTimeRCDPretrainDataset(data_path, filename, split="test", train_ratio=0)
        print(f'Length of dataset: {len(full_test_dataset)}')

        # Limit to max_test_data samples
        if len(full_test_dataset) > max_test_data:
            indices = torch.randperm(len(full_test_dataset))[:max_test_data].tolist()
            test_dataset = torch.utils.data.Subset(full_test_dataset, indices)
            print("random")
        else:
            test_dataset = full_test_dataset

        # Create visualization loader for detailed visualization (one by one)
        vis_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Process one sample at a time for visualization
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )

        # Visualize individual samples (one by one)
        num_visualize = min(num_samples, len(test_dataset))

        vis_iter = iter(vis_loader)

        for i in range(num_visualize):
            try:
                vis_batch = next(vis_iter)

                # Run inference for this single sample
                vis_results = self.predict(vis_batch)

                save_path = None
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"sample_{i + 1}_results.png")

                self.visualize_single_sample(vis_results, sample_idx=0, save_path=save_path)

            except StopIteration:
                break

    def zero_shot(self, data):
        """Run zero-shot inference on the provided data."""
        if len(data)  <= self.win_size:
           self.win_size = len(data)

        test_loader = DataLoader(
            dataset=TimeRCDDataset(data, window_size=self.win_size, stride=self.win_size, normalize=True),
            batch_size=self.batch_size,
            collate_fn=test_collate_fn,
            num_workers=0,
            shuffle=False,)

        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        scores = []
        logits = []
        with torch.no_grad():
            for i, batch in loop:
                # Move data to device
                time_series = batch['time_series'].to(self.device)
                # print("Here is the time series shape: ", time_series.shape)
                # print(f"Here are a sample of dataset after normalization: {time_series[:10, :]}")
                batch_size, seq_len, num_features = time_series.shape
                # 对时间序列标准化
                attention_mask = batch['attention_mask'].to(self.device)
                # print("Here is the attention mask shape: ", attention_mask.shape)
                # print("Here is the attention mask: ", attention_mask)
                # Get embeddings
                local_embeddings = self.model(
                    time_series=time_series,
                    mask=attention_mask)

                # Get anomaly predictions
                anomaly_logits = self.model.anomaly_head(local_embeddings)
                anomaly_logits = torch.mean(anomaly_logits, dim=-2)  # (B, seq_len, 2)
                anomaly_probs = F.softmax(anomaly_logits, dim=-1)[..., 1]  # Probability of anomaly (B, seq_len)
                scores.append(anomaly_probs.cpu().numpy())
                logit = anomaly_logits[..., 1] - anomaly_logits[..., 0]  # Anomaly logits (B, seq_len)
                logits.append(logit.cpu().numpy())
        return scores, logits

    def evaluate(self, time_series, mask):
        with torch.no_grad():
            time_series = time_series.to(self.device)
            mask = mask.to(self.device)
            local_embeddings = self.model(time_series = time_series, mask = mask)

            reconstructed = self.model.reconstruction_head(local_embeddings)  # (B, seq_len, num_features, 1)
            reconstructed = reconstructed.squeeze(-1)

            mask_expand = mask.unsqueeze(-1).expand(-1, -1, reconstructed.shape[-1])

            anomaly_probs = ((reconstructed - time_series) ** 2)[mask_expand]
        return anomaly_probs, reconstructed


    def zero_shot_reconstruct(self, data, visualize=True, data_index=None):
        """Run zero-shot inference on the provided data."""
        if len(data) <= self.win_size:
            self.win_size = len(data)

        test_loader = DataLoader(
            dataset=Dataset_UCR(data, window_size=self.win_size),
            batch_size=self.batch_size,
            # collate_fn=collate_fn,
            num_workers=0,
            shuffle=False, )

        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
        scores = []
        with torch.no_grad():
            for i, (x, mask) in loop:
                # Move data to device
                print("Here is the batch type: ", type(x))
                print("Shape: ", np.array(x).shape)
                time_series = torch.tensor(x, dtype=torch.float32).to(self.device)  # (B, seq_len, num_features)
                mask_tensor = torch.tensor(mask, dtype=torch.bool).to(self.device)
                # print("Here is the time series shape: ", time_series.shape)
                # 对时间序列标准化
                # attention_mask = batch['attention_mask'].to(self.device)
                score, reconstructed = self.evaluate(time_series, mask_tensor)

                scores.append(score)

                # Visualize the first batch if requested
                if visualize:
                    self.visualize_reconstruction(original=time_series[0].cpu().numpy(),
                                                  reconstructed=reconstructed.cpu().numpy(),
                                                  mask=mask_tensor[0].cpu().numpy(),
                                                  scores=score.cpu().numpy(),
                                                  save_path=f"/home/lihaoyang/Huawei/TSB-AD/Synthetic/random_mask_anomaly_head_Time_RCD_Reconstruction_5000/plot/",
                                                  index=data_index)

        return scores

    def visualize_reconstruction(self, original, reconstructed, mask, scores, index, save_path=None):
        """Visualize reconstruction results for a single sample."""
        import matplotlib.pyplot as plt

        seq_len = len(original)
        time_axis = np.arange(seq_len)

        # Squeeze singleton dimensions
        original = original.squeeze()
        reconstructed = reconstructed.squeeze(0).squeeze(-1)
        scores = scores.squeeze()

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # 1. Reconstruction plot
        ax1 = axes[0]
        ax1.plot(time_axis, original, 'b-', label='Original', linewidth=2, alpha=0.8)
        ax1.plot(time_axis, reconstructed, 'r--', label='Reconstructed', linewidth=2, alpha=0.8)

        # Highlight masked regions
        mask_regions = []
        in_mask = False
        start_idx = 0
        for i, is_masked in enumerate(mask):
            if is_masked and not in_mask:
                start_idx = i
                in_mask = True
            elif not is_masked and in_mask:
                mask_regions.append((start_idx, i - 1))
                in_mask = False
        if in_mask:
            mask_regions.append((start_idx, len(mask) - 1))

        for start, end in mask_regions:
            ax1.axvspan(start, end, alpha=0.2, color='red',
                        label='Masked Region' if start == mask_regions[0][0] else "")

        ax1.set_title('Time Series Reconstruction', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Reconstruction error plot
        ax2 = axes[1]
        recon_error = np.abs(original - reconstructed)
        ax2.plot(time_axis, recon_error, 'g-', label='Reconstruction Error', linewidth=2, alpha=0.8)

        # Plot scores if available (mapped to time steps)
        if len(scores) == mask.sum():
            # Scores are only for masked points, map back to full sequence
            full_scores = np.zeros(seq_len)
            full_scores[mask] = scores
            ax2_twin = ax2.twinx()
            ax2_twin.plot(time_axis, full_scores, 'orange', label='Anomaly Scores', linewidth=1.5, alpha=0.7)
            ax2_twin.set_ylabel('Anomaly Score', color='orange')
            ax2_twin.legend(loc='upper right')

        ax2.set_title('Reconstruction Error', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Absolute Error')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, f"reconstruction_sample_{index}_results.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print("Visualization saved to: ", save_path)

        # plt.show()


class Dataset_UCR(Dataset):
    def __init__(self, data, window_size: int = 1000):
        super().__init__()
        self.data = data.reshape(-1, 1) if len(data.shape) == 1 else data
        self.window_size = window_size
        self._load_data()
        self._process_windows()
    
    def _load_data(self):
        # train_data = np.load(train_path, allow_pickle=True)  # (seq_len, num_features)
        # test_data = np.load(test_path, allow_pickle=True)  # (seq_len, num_features)
        # test_labels = np.load(label_path, allow_pickle=True)  # (seq_len, )
        train_data = self.data
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        self.raw_test = scaler.transform(self.data)

    def _process_windows(self):
        if len(self.raw_test) <= self.window_size:
            self.test = np.expand_dims(self.raw_test, axis=0)
            # self.test_labels = np.expand_dims(self.raw_labels, axis=0)
            self.mask = np.expand_dims(np.ones(len(self.raw_test), dtype=bool), axis=0)
        else:
            self.raw_masks = np.ones(len(self.raw_test), dtype=bool)
            padding = self.window_size - (len(self.raw_test) % self.window_size)
            if padding < self.window_size:
                self.raw_test = np.pad(self.raw_test, ((0, padding), (0, 0)), mode='constant')
                # self.raw_labels = np.pad(self.raw_labels, (0, padding), mode='constant')
                self.raw_masks = np.pad(self.raw_masks, (0, padding), mode='constant')
            self.test = self.raw_test.reshape(-1, self.window_size, self.raw_test.shape[1])
            # self.test_labels = self.raw_labels.reshape(-1, self.window_size)
            self.mask = self.raw_masks.reshape(-1, self.window_size)
            assert self.test.shape[0] == self.test_labels.shape[0] == self.mask.shape[0], "Inconsistent window sizes"

    def __len__(self):
        return len(self.test)

    def __getitem__(self, index):
        return np.float32(self.test[index]), self.mask[index]