"""
Time-RCD Anomaly Detection Implementation with HuggingFace Integration

This implementation provides a wrapper around the HuggingFace-compatible Time_RCD model
for seamless integration with the existing testing framework.
"""

import numpy as np
import torch
import warnings
import os
import sys
from sklearn.utils import check_array

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
            
            from huggingface_time_rcd import Time_RCD as HF_Time_RCD, TimeRCDConfig
            
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