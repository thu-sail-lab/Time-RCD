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
class LLMConfig:
    """Configuration for language model.

    Attributes:
        model_name: Name of the base language model.
        max_batch_size: Maximum batch size for processing.
        num_fixed_tokens: Number of fixed tokens in prompt.
        num_prototype: Number of prototypes.
        num_heads: Number of attention heads.
    """
    model_name: str = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
    max_batch_size: int = 20
    num_fixed_tokens: int = 30
    num_prototype: int = 1000
    num_heads: int = 8


@dataclass
class TimeRCDConfig:
    """Configuration class for Time_RCD model.

    This class contains all hyperparameters and settings for the Time_RCD model.
    It is implemented as a dataclass for easy instantiation and modification.

    Attributes:
        ts_config: Configuration for time series encoder.
        llm_config: Configuration for language model.
        batch_size: Training batch size.
        learning_rate: Learning rate for optimization.
        num_epochs: Number of training epochs.
        max_seq_len: Maximum sequence length.
        accumulation_steps: Gradient accumulation steps.
        weight_decay: Weight decay for optimization.
        enable_ts_train: Whether to train the time series encoder.
        data_path: Path to dataset.
        load_data: Whether to load data from checkpoint.
        ts_encoder_path: Path to time series encoder checkpoint.
        time_rcd_path: Path to Time_RCD checkpoint.
        checkpoint_dir: Directory for saving checkpoints.
        log_freq: Logging frequency in steps.
        save_step_freq: Checkpoint saving frequency in steps.
        model_prefix: Prefix for saved model files.
        test_batch_limit: Maximum number of test batches.
        seed: Random seed for reproducibility.
        cuda_devices: CUDA device IDs to use.
        device: Device to run the model on.
    """

    # Model configurations
    ts_config: TimeSeriesConfig = field(default_factory=TimeSeriesConfig)
    llm_config: LLMConfig = field(default_factory=LLMConfig)

    # Configuration file paths
    deepspeed_config_path: str = "experiments/configs/deepspeed_config.json"
    accelerate_config_path: str = "experiments/configs/accelerate_config.yaml"

    # Training parameters
    batch_size: int = 3
    learning_rate: float = 1e-4
    num_epochs: int = 1000
    max_seq_len: int = 512
    dropout: float = 0.1
    accumulation_steps: int = 1
    weight_decay: float = 1e-5
    enable_ts_train: bool = False
    # enable_anomaly_detection_head: bool = False

    # Data parameters
    data_path: str = "/home3/lantian/ChatTS/data/ChatTSQA/time_rcd_dataset_20250614_133353"
    test_data_path: str = "/home3/lantian/ChatTS/data/ChatTSQA/time_rcd_test_dataset_20250621_090751"
    pretrain_data_path: str = "/home2/lijinbo/Projects/dataset/ChatTS/result"

    # Load data
    load_data: bool = True
    load_pretrain_path: str = "experiments/checkpoints/pretrain/pretrain_checkpoint_latest.pth"
    load_path: str = "experiments/checkpoints/checkpoint_step.pth"
    # ts_encoder_path: str = "experiments/checkpoints/checkpoint_step.pth"
    # time_rcd_path: str = "experiments/checkpoints/time_rcd_qa.pt"

    # Logging and checkpoint parameters
    checkpoint_dir: str = "experiments/checkpoints"
    log_freq: int = 100
    save_freq: int = 10
    save_step_freq: int = 100
    model_prefix: str = "time_rcd_qa_by_pretrain"

    # Evaluation parameters
    test_batch_limit: int = 20
    early_stopping_patience: int = 7

    # Random seed
    seed: int = 72

    # CUDA settings
    cuda_devices: str = "0, 1, 2, 3"
    dist_port: str = "12355"  # Port for distributed training communication
    device: str = "cuda"
    use_multi_gpu: bool = True
    mixed_precision: bool = True

    def to_dict(self) -> Dict[str, any]:
        return {
            "ts_config": self.ts_config.__dict__,
            "llm_config": self.llm_config.__dict__,
            "deepspeed_config_path": self.deepspeed_config_path,
            "accelerate_config_path": self.accelerate_config_path,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "max_seq_len": self.max_seq_len,
            "seed": self.seed,
            "data_path": self.data_path,
            "checkpoint_dir": self.checkpoint_dir,
            "log_freq": self.log_freq,
            "test_batch_limit": self.test_batch_limit,
            "save_step_freq": self.save_step_freq,
            "accumulation_steps": self.accumulation_steps,
            "model_prefix": self.model_prefix,
            "device": self.device,
        }

default_config = TimeRCDConfig()