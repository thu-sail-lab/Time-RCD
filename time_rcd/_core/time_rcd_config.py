from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TimeSeriesConfig:
    """Configuration for time series encoder.

    Attributes:
        d_model: Dimension of model hidden states.
        d_proj: Dimension of projection layer.
        patch_size: Size of time series patches.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        d_ff_dropout: Dropout rate for feed-forward networks.
        use_rope: Whether to use Rotary Position Embedding.
        activation: Activation function name.
        num_features: Number of input features.
    """
    d_model: int = 512
    d_proj: int = 256
    patch_size: int = 4
    num_query_tokens: int = 1
    num_layers: int = 8
    num_heads: int = 8
    d_ff_dropout: float = 0.1
    use_rope: bool = True
    activation: str = "gelu"
    num_features: int = 1


@dataclass
class TimeRCDConfig:
    """Configuration class for Time_RCD model.

    This class contains all hyperparameters and settings for the Time_RCD model.
    It is implemented as a dataclass for easy instantiation and modification.

    Attributes:
        ts_config: Configuration for time series encoder.
        batch_size: Training batch size.
        learning_rate: Learning rate for optimization.
        num_epochs: Number of training epochs.
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate.
        accumulation_steps: Gradient accumulation steps.
        weight_decay: Weight decay for optimization.
        enable_ts_train: Whether to train the time series encoder.
        seed: Random seed for reproducibility.
    """

    # Model configurations
    ts_config: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)

    # Training parameters
    batch_size: int = 3
    learning_rate: float = 1e-4
    num_epochs: int = 1000
    max_seq_len: int = 512
    dropout: float = 0.1
    accumulation_steps: int = 1
    weight_decay: float = 1e-5
    enable_ts_train: bool = False
    seed: int = 72

    def to_dict(self) -> Dict[str, any]:
        return {
            "ts_config": self.ts_config.__dict__,
        }

default_config = TimeRCDConfig()