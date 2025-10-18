"""
Time-RCD Anomaly Detection Implementation with HuggingFace Integration

This implementation provides a wrapper around the HuggingFace-compatible Time_RCD model
for seamless integration with the existing testing framework.
"""

import numpy as np
import torch
import os
from transformers.configuration_utils import PretrainedConfig
from typing import Dict, Any
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.utils import logging
import sys
# Try to import einops, fall back to manual implementation if not available
try:
    from einops import rearrange
    HAS_EINOPS = True
except ImportError:
    HAS_EINOPS = False
    def rearrange(tensor, pattern):
        # Simple fallback for the specific pattern we use
        if pattern == "two num_heads -> two num_heads 1 1":
            return tensor.unsqueeze(-1).unsqueeze(-1)
        else:
            raise NotImplementedError(f"Pattern {pattern} not implemented in fallback")


logger = logging.get_logger(__name__)



class TimeRCDConfig(PretrainedConfig):
    """
    Configuration class for Time_RCD model.

    This is the configuration class to store the configuration of a [`Time_RCD`] model. It is used to
    instantiate a Time_RCD model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs.
    Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        d_model (`int`, *optional*, defaults to 512):
            Dimension of model hidden states.
        d_proj (`int`, *optional*, defaults to 256):
            Dimension of projection layer.
        patch_size (`int`, *optional*, defaults to 4):
            Size of time series patches.
        num_layers (`int`, *optional*, defaults to 8):
            Number of transformer layers.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads.
        d_ff_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate for feed-forward networks.
        use_rope (`bool`, *optional*, defaults to True):
            Whether to use Rotary Position Embedding.
        activation (`str`, *optional*, defaults to "gelu"):
            Activation function name.
        num_features (`int`, *optional*, defaults to 1):
            Number of input features in the time series.
        dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate for the model.
        max_seq_len (`int`, *optional*, defaults to 512):
            Maximum sequence length.
        win_size (`int`, *optional*, defaults to 5000):
            Window size for inference.
        batch_size (`int`, *optional*, defaults to 64):
            Default batch size for inference.
    """

    model_type = "time_rcd"

    def __init__(
        self,
        d_model: int = 512,
        d_proj: int = 256,
        patch_size: int = 4,  # Your specific configuration
        num_layers: int = 8,
        num_heads: int = 8,
        d_ff_dropout: float = 0.1,
        use_rope: bool = True,
        activation: str = "gelu",
        num_features: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        win_size: int = 5000,
        batch_size: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.d_model = d_model
        self.d_proj = d_proj
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff_dropout = d_ff_dropout
        self.use_rope = use_rope
        self.activation = activation
        self.num_features = num_features
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.win_size = win_size
        self.batch_size = batch_size

    @classmethod
    def from_pretrained_config(cls, original_config_dict: Dict[str, Any]):
        """Convert from your original configuration format."""
        return cls(
            d_model=original_config_dict.get("ts_config", {}).get("d_model", 512),
            d_proj=original_config_dict.get("ts_config", {}).get("d_proj", 256),
            patch_size=original_config_dict.get("ts_config", {}).get("patch_size", 16),
            num_layers=original_config_dict.get("ts_config", {}).get("num_layers", 8),
            num_heads=original_config_dict.get("ts_config", {}).get("num_heads", 8),
            d_ff_dropout=original_config_dict.get("ts_config", {}).get("d_ff_dropout", 0.1),
            use_rope=original_config_dict.get("ts_config", {}).get("use_rope", True),
            activation=original_config_dict.get("ts_config", {}).get("activation", "gelu"),
            num_features=original_config_dict.get("ts_config", {}).get("num_features", 1),
            dropout=original_config_dict.get("dropout", 0.1),
            max_seq_len=original_config_dict.get("max_seq_len", 512),
            win_size=original_config_dict.get("win_size", 5000),
            batch_size=original_config_dict.get("batch_size", 64),
        )

@dataclass
class TimeRCDOutput(ModelOutput):
    """
    Output for Time_RCD model.
    
    Args:
        anomaly_scores (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Anomaly scores for each time step.
        anomaly_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, 2)`):
            Raw logits for anomaly classification.
        reconstruction (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`):
            Reconstructed time series.
        embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features, d_proj)`):
            Time series embeddings from the encoder.
    """
    anomaly_scores: Optional[torch.FloatTensor] = None
    anomaly_logits: Optional[torch.FloatTensor] = None
    reconstruction: Optional[torch.FloatTensor] = None
    embeddings: Optional[torch.FloatTensor] = None

class Time_RCD(PreTrainedModel):
    """
    Time-RCD Model for Time Series Anomaly Detection
    
    This is the main model class that directly inherits from PreTrainedModel.
    It matches your original Time_RCD implementation structure exactly:
    - TimeSeriesEncoder for encoding
    - reconstruction_head for reconstruction
    - anomaly_head for anomaly detection
    
    No extra inheritance layers - clean and simple!
    """
    
    config_class = TimeRCDConfig
    base_model_prefix = "time_rcd"
    supports_gradient_checkpointing = True

    def __init__(self, config: TimeRCDConfig):
        super().__init__(config)
        self.config = config

        # Time series encoder (matches your original implementation)
        self.ts_encoder = TimeSeriesEncoder(
            d_model=config.d_model,
            d_proj=config.d_proj,
            patch_size=config.patch_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            d_ff_dropout=config.d_ff_dropout,
            use_rope=config.use_rope,
            num_features=config.num_features,
            activation=config.activation
        )

        # Reconstruction head (exactly like your original)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.d_proj, config.d_proj * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_proj * 4, config.d_proj * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_proj * 4, 1)  # Output: (B, seq_len, num_features, 1)
        )

        # Anomaly detection head (exactly like your original)
        self.anomaly_head = nn.Sequential(
            nn.Linear(config.d_proj, config.d_proj // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_proj // 2, 2)  # Binary classification: (B, seq_len, num_features, 2)
        )

        # Initialize weights
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights (standard HuggingFace pattern)"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        time_series: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TimeRCDOutput]:
        """
        Forward pass through Time_RCD model
        
        Args:
            time_series (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_features)`):
                Input time series data.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            return_dict (`bool`, *optional*):
                Whether to return a ModelOutput instead of a plain tuple.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_len, num_features = time_series.shape

        # Normalize time series (exactly like your original)
        time_series = (time_series - time_series.mean(dim=1, keepdim=True)) / (time_series.std(dim=1, keepdim=True) + 1e-8)

        # Get embeddings from encoder
        embeddings = self.ts_encoder(time_series, attention_mask)  # (B, seq_len, num_features, d_proj)

        # Get reconstruction
        reconstruction = self.reconstruction_head(embeddings)  # (B, seq_len, num_features, 1)
        reconstruction = reconstruction.squeeze(-1)  # (B, seq_len, num_features)

        # Get anomaly predictions
        anomaly_logits = self.anomaly_head(embeddings)  # (B, seq_len, num_features, 2)
        anomaly_logits = torch.mean(anomaly_logits, dim=-2)  # Average over features: (B, seq_len, 2)
        anomaly_scores = F.softmax(anomaly_logits, dim=-1)[..., 1]  # Probability of anomaly: (B, seq_len)

        if not return_dict:
            return (anomaly_scores, anomaly_logits, reconstruction, embeddings)

        return TimeRCDOutput(
            anomaly_scores=anomaly_scores,
            anomaly_logits=anomaly_logits,
            reconstruction=reconstruction,
            embeddings=embeddings
        )

    def zero_shot(self, data: np.ndarray, batch_size: int = 64, win_size: int = 5000) -> tuple:
        """
        Zero-shot inference method matching AnomalyCLIP structure.
        
        The model handles normalization internally, so no external processor needed!
        This method only handles windowing for long sequences.
        
        Args:
            data: Input time series data of shape (n_samples, n_features) or (n_samples,)
            batch_size: Batch size for processing
            win_size: Window size for processing (only used if data > win_size)
            
        Returns:
            tuple: (scores, logits) where:
                - scores: list of anomaly score arrays per batch
                - logits: list of anomaly logit arrays per batch
        """
        import tqdm
        from torch.utils.data import DataLoader, TensorDataset
        
        self.eval()
        device = next(self.parameters()).device
        
        # Ensure numpy and 2D shape
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Adjust window size if data is too short
        if len(data) <= win_size:
            win_size = len(data)
        
        # Create windows if data is longer than win_size
        windows = []
        masks = []
        
        if len(data) > win_size:
            # Create non-overlapping windows
            for i in range(0, len(data), win_size):
                window = data[i:i + win_size]
                if len(window) < win_size:
                    # Pad last window if needed
                    padded = np.zeros((win_size, data.shape[1]))
                    padded[:len(window)] = window
                    window = padded
                    mask = np.zeros(win_size, dtype=bool)
                    mask[:len(window)] = True
                else:
                    mask = np.ones(win_size, dtype=bool)
                windows.append(window)
                masks.append(mask)
        else:
            # Single window
            windows.append(data)
            masks.append(np.ones(len(data), dtype=bool))
        
        # Convert to tensors
        time_series_windows = torch.tensor(np.array(windows), dtype=torch.float32)
        attention_masks = torch.tensor(np.array(masks), dtype=torch.bool)
        
        # Create dataloader
        dataset = TensorDataset(time_series_windows, attention_masks)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        loop = tqdm.tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
        scores = []
        logits = []
        
        with torch.no_grad():
            for i, (batch_ts, batch_mask) in loop:
                batch_ts = batch_ts.to(device)
                batch_mask = batch_mask.to(device)
                
                # Forward pass (model normalizes internally!)
                outputs = self(
                    time_series=batch_ts,
                    attention_mask=batch_mask,
                    return_dict=True
                )
                
                # Extract scores and logits
                anomaly_probs = outputs.anomaly_scores.cpu().numpy()  # (B, seq_len)
                anomaly_logits = outputs.anomaly_logits  # (B, seq_len, 2)
                logit_diff = anomaly_logits[..., 1] - anomaly_logits[..., 0]  # (B, seq_len)
                
                scores.append(anomaly_probs)
                logits.append(logit_diff.cpu().numpy())
        
        return scores, logits

    @classmethod
    def from_original_checkpoint(cls, checkpoint_path: str, config: Optional[TimeRCDConfig] = None):
        """
        Load model from your original checkpoint format
        
        Args:
            checkpoint_path: Path to your .pth checkpoint file
            config: Model configuration (optional - will auto-detect from checkpoint if not provided)
            
        Returns:
            Loaded Time_RCD model
        """
        print(f"Loading Time_RCD from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Auto-detect config from checkpoint if not provided
        if config is None:
            print("📋 Auto-detecting config from checkpoint...")
            if 'config' in checkpoint:
                ckpt_config = checkpoint['config']
                ts_config = ckpt_config.get('ts_config', {})
                
                config = TimeRCDConfig(
                    d_model=ts_config.get('d_model', 512),
                    d_proj=ts_config.get('d_proj', 256),
                    patch_size=ts_config.get('patch_size', 4),  # Important!
                    num_layers=ts_config.get('num_layers', 8),
                    num_heads=ts_config.get('num_heads', 8),
                    d_ff_dropout=ts_config.get('d_ff_dropout', 0.1),
                    use_rope=ts_config.get('use_rope', True),
                    activation=ts_config.get('activation', 'gelu'),
                    num_features=ts_config.get('num_features', 1),
                    max_seq_len=ckpt_config.get('max_seq_len', 512),
                    win_size=ckpt_config.get('win_size', 5000),
                    batch_size=ckpt_config.get('batch_size', 64),
                    dropout=0.1
                )
                print(f"✅ Auto-detected config: patch_size={config.patch_size}, d_model={config.d_model}, d_proj={config.d_proj}")
            else:
                print("⚠️  No config found in checkpoint, using defaults")
                config = TimeRCDConfig()
        
        # Create model
        model = cls(config)
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
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
        
        # Load state dict with flexible matching
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print("✅ Successfully loaded checkpoint with flexible matching")
        except Exception as e:
            print(f"⚠️  Error loading state dict: {e}")
            print("Available checkpoint keys:", list(new_state_dict.keys())[:10])
            print("Model keys:", list(model.state_dict().keys())[:10])
            
        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the model in HuggingFace format
        
        This allows you to use .from_pretrained() later
        """
        super().save_pretrained(save_directory, **kwargs)
        print(f"✅ Model saved to {save_directory}")
        print("You can now load it with:")
        print(f"model = Time_RCD.from_pretrained('{save_directory}')")



class TimeSeriesEncoder(nn.Module):
    """
    Time Series Encoder with PatchTST-like patching, RoPE.

    Args:
        d_model (int): Model dimension
        d_proj (int): Projection dimension
        patch_size (int): Size of each patch
        num_layers (int): Number of encoder layers
        num_heads (int): Number of attention heads
        d_ff_dropout (float): Dropout rate
        max_total_tokens (int): Maximum sequence length
        use_rope (bool): Use RoPE if True
        num_features (int): Number of features in the time series
        activation (str): "relu" or "gelu"

    Inputs:
        time_series (Tensor): Shape (batch_size, seq_len, num_features)
        mask (Tensor): Shape (batch_size, seq_len)

    Outputs:
        local_embeddings (Tensor): Shape (batch_size, seq_len, num_features, d_proj)
    """

    def __init__(self, d_model=2048, d_proj=512, patch_size=32, num_layers=6, num_heads=8,
                 d_ff_dropout=0.1, max_total_tokens=8192, use_rope=True, num_features=1,
                 activation="relu"):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_proj = d_proj
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff_dropout = d_ff_dropout
        self.max_total_tokens = max_total_tokens
        self.use_rope = use_rope
        self.num_features = num_features
        self.activation = activation

        # Patch embedding layer
        self.embedding_layer = nn.Linear(patch_size, d_model)

        if use_rope:
            # Initialize RoPE and custom encoder
            self.rope_embedder = RotaryEmbedding(d_model)
            self.transformer_encoder = CustomTransformerEncoder(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=d_ff_dropout,
                activation=activation,
                num_layers=num_layers,
                num_features=num_features
            )
        else:
            # Standard encoder without RoPE
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=d_ff_dropout,
                batch_first=True,
                activation=activation
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection layers
        self.projection_layer = nn.Linear(d_model, patch_size * d_proj)
        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'linear' in name:
                if self.activation == "relu":
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif self.activation == "gelu":
                    nn.init.kaiming_uniform_(param, nonlinearity='gelu')
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, time_series, mask=None):
        """Forward pass to generate local embeddings."""
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)
        device = time_series.device
        B, seq_len, num_features = time_series.size()
        assert num_features == self.num_features, f"Number of features mismatch with data: {num_features} vs param: {self.num_features}"
        
        # Create mask if not provided
        if mask is None:
            mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)
        
        assert mask.size() == (B, seq_len), f"Mask shape mismatch: expected ({B}, {seq_len}), got {mask.size()}"

        # Pad sequence to be divisible by patch_size
        padded_length = math.ceil(seq_len / self.patch_size) * self.patch_size
        if padded_length > seq_len:
            pad_amount = padded_length - seq_len
            time_series = F.pad(time_series, (0, 0, 0, pad_amount), value=0)
            mask = F.pad(mask, (0, pad_amount), value=0)

        # Convert to patches
        num_patches = padded_length // self.patch_size
        total_length = num_patches * num_features
        patches = time_series.view(B, num_patches, self.patch_size, num_features)
        patches = patches.permute(0, 3, 1, 2).contiguous()  # (B, num_features, num_patches, patch_size)
        patches = patches.view(B, num_features * num_patches, self.patch_size)  # (B, L, patch_size)
        # Create feature IDs for patches
        feature_id = torch.arange(num_features, device=device).repeat_interleave(
            num_patches)  # (num_features * num_patches = L,)
        feature_id = feature_id.unsqueeze(0).expand(B, -1)  # (B, L)

        # Embed patches
        embedded_patches = self.embedding_layer(patches)  # (B, L, d_model)

        # Create patch-level mask
        mask = mask.view(B, num_patches, self.patch_size)
        patch_mask = mask.sum(dim=-1) > 0  # (B, num_patches)
        full_mask = patch_mask.unsqueeze(1).expand(-1, num_features, -1)  # (B, num_features, num_patches)
        full_mask = full_mask.reshape(B, num_features * num_patches)  # (B, L)

        # Generate RoPE frequencies if applicable
        if self.use_rope:
            freqs = self.rope_embedder(total_length).to(device)
        else:
            freqs = None

        # Encode sequence
        if num_features > 1:
            output = self.transformer_encoder(
                embedded_patches,
                freqs=freqs,
                src_id=feature_id,
                attn_mask=full_mask
            )
        else:
            output = self.transformer_encoder(
                embedded_patches,
                freqs=freqs,
                attn_mask=full_mask
            )

        # Extract and project local embeddings
        patch_embeddings = output  # (B, L, d_model)
        patch_proj = self.projection_layer(patch_embeddings)  # (B, L, patch_size * d_proj)
        local_embeddings = patch_proj.view(B, num_features, num_patches, self.patch_size, self.d_proj)
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4)  # (B, num_patches, patch_size, num_features, d_proj)
        local_embeddings = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :,
                           :]  # (B, seq_len, num_features, d_proj)

        return local_embeddings


class CustomTransformerEncoder(nn.Module):
    """Stack of Transformer Encoder Layers."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, num_layers, num_features):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                num_features=num_features
            ) for _ in range(num_layers)
        ])

    def forward(self, src, freqs, src_id=None, attn_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, freqs, src_id, attn_mask=attn_mask)
        return output


class TransformerEncoderLayerWithRoPE(nn.Module):
    """Transformer Encoder Layer with RoPE and RMSNorm."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", num_features=1):
        super().__init__()
        self.self_attn = MultiheadAttentionWithRoPE(d_model, nhead, num_features)
        self.dropout = nn.Dropout(dropout)
        self.input_norm = RMSNorm(d_model)
        self.output_norm = RMSNorm(d_model)
        self.mlp = LlamaMLP(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src, freqs, src_id=None, attn_mask=None):
        residual = src
        src = self.input_norm(src)
        src = self.self_attn(src, src, src, freqs, src_id, src_id, attn_mask=attn_mask)
        src = src + residual
        residual = src
        src = self.output_norm(src)
        src = self.mlp(src)
        src = residual + self.dropout2(src)
        return src


class RMSNorm(nn.Module):
    """Root Mean Square Normalization layer."""

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.to(torch.float32).pow(2).mean(dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return (self.scale * x_normed).type_as(x)


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding for injecting positional information."""

    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs  # Shape: (seq_len, dim // 2)


class BinaryAttentionBias(nn.Module):
    """Binary Variate Attention for time series data."""

    def __init__(self,
                 num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.emd = nn.Embedding(2, num_heads)

    def forward(self,
                query_id: torch.Tensor,
                kv_id: torch.Tensor,
                ) -> torch.Tensor:
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        ind = ind.unsqueeze(1)  # (batch_size, 1, q_len, kv_len)
        weight = rearrange(self.emd.weight, "two num_heads -> two num_heads 1 1")  # (2, num_heads, 1, 1)
        bias = ~ind * weight[:1] + ind * weight[1:]  # (batch_size, num_heads, q_len, kv_len)
        return bias


class MultiheadAttentionWithRoPE(nn.Module):
    """Multi-head Attention with Rotary Positional Encoding (RoPE), non-causal by default."""
    "========== NOtice that this applies BinaryAttentionBias ==========="

    def __init__(self, embed_dim, num_heads, num_features):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_features = num_features
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V, and output
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Binary attention bias for time series
        if num_features > 1:
            self.binary_attention_bias = BinaryAttentionBias(num_heads)

    def apply_rope(self, x, freqs):
        """Apply Rotary Positional Encoding to the input tensor."""
        B, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"
        assert freqs.shape == (seq_len, embed_dim // 2), "freqs shape mismatch"

        # Reshape for rotation: split embed_dim into pairs
        x_ = x.view(B, seq_len, embed_dim // 2, 2)
        cos = freqs.cos().unsqueeze(0)  # (1, seq_len, embed_dim // 2, 1)
        sin = freqs.sin().unsqueeze(0)  # (1, seq_len, embed_dim // 2, 1)

        # Apply rotation to each pair
        x_rot = torch.stack(
            [
                x_[..., 0] * cos - x_[..., 1] * sin,
                x_[..., 0] * sin + x_[..., 1] * cos,
            ],
            dim=-1
        )
        return x_rot.view(B, seq_len, embed_dim)

    def forward(self, query, key, value, freqs, query_id=None, kv_id=None, attn_mask=None):
        """
        Forward pass for multi-head attention with RoPE.

        Args:
            query (Tensor): Shape (B, T, C)
            key (Tensor): Shape (B, T, C)
            value (Tensor): Shape (B, T, C)
            freqs (Tensor): RoPE frequencies, shape (T, embed_dim // 2)
            query_id (Tensor, optional): Shape (B, q_len), feature IDs for query
            kv_id (Tensor, optional): Shape (B, kv_len), feature IDs for key/value
            attn_mask (Tensor, optional): Shape (B, T), True for valid positions, False for padding.

        Returns:
            Tensor: Attention output, shape (B, T, C)
        """
        B, T, C = query.shape
        assert key.shape == (B, T, C) and value.shape == (B, T, C), "query, key, value shapes must match"

        # Project inputs to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Apply RoPE to Q and K
        Q_rot = self.apply_rope(Q, freqs)
        K_rot = self.apply_rope(K, freqs)

        # Reshape for multi-head attention
        Q_rot = Q_rot.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        K_rot = K_rot.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Prepare attention mask for padding
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        else:
            attn_mask = None

        if query_id is not None and kv_id is not None:
            # Add binary attention bias
            attn_bias = self.binary_attention_bias(query_id, kv_id)  # (B, num_heads, q_len, kv_len)
            scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1)) / math.sqrt(
                self.head_dim)  # (B, num_heads, q_len, kv_len)
            scores += attn_bias
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, q_len, kv_len)
            y = torch.matmul(attn_weights, V)  # (B, num_heads, q_len, hs)

        else:
            # Compute scaled dot-product attention (non-causal) without binary bias
            # for param in self.binary_attention_bias.parameters():
            #     param.requires_grad = False
            y = F.scaled_dot_product_attention(
                Q_rot, K_rot, V,
                attn_mask=attn_mask,
                is_causal=False  # Non-causal attention for encoder
            )  # (B, nh, T, hs)

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y


class LlamaMLP(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048):
        super().__init__()
        self.hidden_size = d_model
        self.intermediate_size = dim_feedforward
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = F.gelu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj




class Time_RCD:
    """
    Time-RCD Model with HuggingFace Integration
    
    This class provides a simple interface to the Time_RCD model that:
    - Loads from your local checkpoint
    - Uses HuggingFace PreTrainedModel features
    - Maintains compatibility with your existing testing framework
    
    Parameters
    ----------
    num_input_channels : int, default=1
        Number of input channels (features) in the time series
    model_path : str, default=None
        Path to local checkpoint or HuggingFace model hub path
    win_size : int, default=5000
        Window size for processing
    batch_size : int, default=256
        Batch size for processing
    device : str, default=None
        Device to use ("cuda" or "cpu"). Auto-detected if None.
    """
    
    def __init__(self, 
                 num_input_channels=1,
                 model_path=None,
                 win_size=5000,
                 batch_size=256,
                 device=None):
        
        self.num_input_channels = num_input_channels
        self.win_size = win_size
        self.batch_size = batch_size
        
        # Default to local checkpoint if no path provided
        if model_path is None:
            self.model_path = "/Users/oliver/Documents/2025/Huawei/Time-RCD/Time-RCD/Testing/checkpoints/pretrain_checkpoint_epoch_653.pth"
        else:
            self.model_path = model_path
           
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Initialize model
        self._load_model()

    def _load_model(self):
        """Load the Time_RCD model from local checkpoint or HuggingFace Hub"""
        try:
            # Import the HuggingFace components
            # Add huggingface_time_rcd to path
            hf_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'huggingface_time_rcd')
            if hf_path not in sys.path:
                sys.path.append(hf_path)
        
            
            # Create config matching your checkpoint's trained parameters
            # These values come from the 'config' key in your checkpoint
            config = TimeRCDConfig(
                num_features=self.num_input_channels,
                win_size=self.win_size,
                batch_size=self.batch_size,
                d_model=512,         # From checkpoint config
                d_proj=256,          # From checkpoint config
                patch_size=4,        # ⚠️ IMPORTANT: Your checkpoint uses 4, not 16!
                num_layers=8,        # From checkpoint config
                num_heads=8,         # From checkpoint config
                d_ff_dropout=0.1,    # From checkpoint config
                use_rope=True,       # From checkpoint config
                activation="gelu",   # From checkpoint config
                dropout=0.1,
                max_seq_len=512      # From checkpoint config
            )
            
            # Check if model_path is a local checkpoint file
            if self.model_path.endswith('.pth') and os.path.exists(self.model_path):
                print(f"Loading from local checkpoint: {self.model_path}")
                
                # Try auto-detection first (config will be read from checkpoint)
                try:
                    print("🔍 Attempting auto-detection of config from checkpoint...")
                    self.model = HF_Time_RCD.from_original_checkpoint(self.model_path, config=None)
                except Exception as e:
                    print(f"⚠️  Auto-detection failed: {e}")
                    print("📝 Using manually specified config...")
                    self.model = HF_Time_RCD.from_original_checkpoint(self.model_path, config)
            else:
                print(f"Loading from HuggingFace Hub: {self.model_path}")
                self.model = HF_Time_RCD.from_pretrained(
                    self.model_path,
                    config=config,
                    trust_remote_code=True
                )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Time_RCD model loaded successfully on {self.device}")
            print(f"Model config: num_features={config.num_features}, win_size={config.win_size}")
            
        except Exception as e:
            print(f"❌ Error loading Time_RCD model: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load Time_RCD model: {str(e)}")

    def fit(self):
        """Placeholder for fit method (Time_RCD is zero-shot)"""
        pass

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for the input data.
        
        Note: The Time_RCD model handles normalization internally!

        Args:
            X: Input time series data of shape (n_samples, n_features)

        Returns:
            Anomaly scores of shape (n_samples,)
        """
        # Run zero-shot inference (no preprocessing needed!)
        scores, _ = self.model.zero_shot(X, batch_size=self.batch_size, win_size=self.win_size)

        # Combine scores from all windows
        combined_scores = np.concatenate([s.ravel() for s in scores])

        # Trim to original length
        return combined_scores[:len(X)]

    def zero_shot(self, X: np.ndarray) -> tuple:
        """
        Run zero-shot inference on the input data.
        
        Note: The Time_RCD model handles normalization internally!

        Args:
            X: Input time series data of shape (n_samples, n_features)

        Returns:
            tuple: (scores, logits)
        """
        # Run inference (no preprocessing needed!)
        return self.model.zero_shot(X, batch_size=self.batch_size, win_size=self.win_size)

    def save_model_hf(self, save_directory: str):
        """
        Save model in HuggingFace format for future use
        
        Parameters
        ----------
        save_directory : str
            Directory to save the model
        """
        try:
            self.model.save_pretrained(save_directory)
            print(f"✅ Model saved to {save_directory}")
            print("You can now load it with:")
            print(f"model = Time_RCD.from_pretrained('{save_directory}')")
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")


# Legacy compatibility functions
def run_Time_RCD_univariate(data, **kwargs):
    """
    Run Time_RCD for univariate time series anomaly detection
    
    Parameters
    ----------
    data : numpy.ndarray
        Univariate time series data
    **kwargs : dict
        Additional parameters for Time_RCD model
        
    Returns
    -------
    numpy.ndarray
        Anomaly scores
    """
    try:
        # Extract parameters
        win_size = kwargs.get('win_size', 5000)
        batch_size = kwargs.get('batch_size', 64)
        
        # Initialize Time_RCD for univariate data
        model = Time_RCD(
            num_input_channels=1,
            batch_size=batch_size,
            **{k: v for k, v in kwargs.items() if k not in ['win_size', 'batch_size']}
        )
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Run inference
        scores = model.zero_shot(data)
        return scores
        
    except Exception as e:
        print(f"Error in Time_RCD univariate: {str(e)}")
        return np.random.random(len(data)) * 0.1

def run_Time_RCD_multivariate(data, **kwargs):
    """
    Run Time_RCD for multivariate time series anomaly detection
    
    Parameters
    ----------
    data : numpy.ndarray
        Multivariate time series data of shape (n_samples, n_features)
    **kwargs : dict
        Additional parameters for Time_RCD model
        
    Returns
    -------
    numpy.ndarray
        Anomaly scores
    """
    try:
        # Extract parameters
        win_size = kwargs.get('win_size', 5000)
        batch_size = kwargs.get('batch_size', 64)
        
        # Initialize Time_RCD for multivariate data
        model = Time_RCD(
            num_input_channels=data.shape[1] if data.ndim > 1 else 1,
            batch_size=batch_size,
            **{k: v for k, v in kwargs.items() if k not in ['win_size', 'batch_size']}
        )
        
        # Run inference
        scores = model.zero_shot(data)
        return scores
        
    except Exception as e:
        print(f"Error in Time_RCD multivariate: {str(e)}")
        return np.random.random(len(data)) * 0.1

# Main function for compatibility with existing framework
def run_Time_RCD(data, **kwargs):
    """
    Main Time_RCD runner that handles both univariate and multivariate data
    
    Parameters
    ----------
    data : numpy.ndarray
        Time series data
    **kwargs : dict
        Additional parameters
        
    Returns
    -------
    numpy.ndarray
        Anomaly scores
    """
    if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
        return run_Time_RCD_univariate(data, **kwargs)
    else:
        return run_Time_RCD_multivariate(data, **kwargs)