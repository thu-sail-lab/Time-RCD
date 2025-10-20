import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import tqdm
import os
from transformers import AutoTokenizer
from typing import Optional, Tuple

# Add debugging prints to understand the import issue
import sys

# print(f"Python path: {sys.path}")
# print(f"Current working directory: {os.getcwd()}")
# print(f"Current file location: {__file__}")
# print(f"Current file directory: {os.path.dirname(__file__)}")
#
# # Check if the utils directory exists
# utils_path = os.path.join(os.path.basename(os.path.dirname(__file__)), "utils")
# print(f"Utils path: {utils_path}")
# print(f"Utils directory exists: {os.path.exists(utils_path)}")
# print(f"Utils directory contents: {os.listdir(utils_path) if os.path.exists(utils_path) else 'Directory not found'}")
#
# # Check if dataset.py exists
# dataset_path = os.path.join(utils_path, "dataset.py")
# print(f"Dataset file path: {dataset_path}")
# print(f"Dataset file exists: {os.path.exists(dataset_path)}")

# Try different import approaches

os.chdir("/home/lihaoyang/Huawei/TSB-AD/TSB_AD")

try:
    from utils.dataset import ReconstructDataset

    print("Relative import successful")
except ImportError as e:
    print(f"Relative import failed: {e}")

    # Try absolute import
    try:
        from TSB_AD.utils.dataset import ReconstructDataset

        print("Absolute import successful")
    except ImportError as e2:
        print(f"Absolute import failed: {e2}")

        # Try adding parent directory to path
        try:
            parent_dir = os.path.dirname(os.path.dirname(__file__))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from utils.dataset import ReconstructDataset

            print("Import with modified path successful")
        except ImportError as e3:
            print(f"Import with modified path failed: {e3}")

from .base import BaseDetector


# ...existing code...

class Time_MOE(BaseDetector):
    def __init__(self, device, args=None, win_size=64, batch_size=32):
        self.win_size = win_size
        self.batch_size = batch_size
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "Maple728/TimeMoE-50M", device_map=self.device, trust_remote_code=True,
        )
        return model

    # def _acquire_device(self):
    #     if True:
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(
    #             self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
    #         device = torch.device('cuda:{}'.format(self.args.gpu))
    #         print('Use GPU: cuda:{}'.format(self.args.gpu))
    #     else:
    #         device = torch.device('cpu')
    #         print('Use CPU')
    #     return device

    def decision_function(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def fit(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None) -> None:
        pass

    def zero_shot(self, data):
        test_loader = DataLoader(
            dataset=ReconstructDataset(data, window_size=self.win_size, stride=self.win_size, normalize=True),
            batch_size=self.batch_size,
            shuffle=False)

        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

        test_scores = []
        test_labels = []
        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            for i, (batch_x, batch_y) in loop:
                batch_x = batch_x.float().to(self.device)
                # print(f"Batch {i} - batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}")
                # print("Here is the batch_x:", batch_x[:10])
                # Reshape batch_x to match model expectations
                # TimeMoE expects 2D input: (batch_size, sequence_length)
                if batch_x.dim() == 3:
                    # If input is (batch_size, sequence_length, features), flatten features
                    batch_x = batch_x.reshape(batch_x.shape[0], -1)
                    # print(f"Batch {i} - reshaped batch_x to 2D: {batch_x.shape}")
                elif batch_x.dim() > 3:
                    # If more dimensions, flatten to 2D
                    batch_x = batch_x.reshape(batch_x.shape[0], -1)
                    # print(f"Batch {i} - reshaped batch_x to 2D: {batch_x.shape}")

                # Ensure batch_x is 2D and convert to long tensor for token generation
                if batch_x.dim() == 1:
                    batch_x = batch_x.unsqueeze(0)
                    # print(f"Batch {i} - batch_x was 1D, reshaped to 2D: {batch_x.shape}")

                # Convert to integer tokens if needed (TimeMoE might expect discrete tokens)
                # For time series, we might need to discretize or use the model differently
                try:
                    # Try direct generation first
                    score = self.model.generate(batch_x.long(), max_new_tokens=self.win_size)
                    score = score[:, -self.win_size:]
                except Exception as e:
                    print(f"Generation failed with long tensor, trying with float: {e}")
                    try:
                        # If that fails, try with original float tensor but ensure 2D
                        score = self.model.generate(batch_x, max_new_tokens=self.win_size)
                        score = score[:, -self.win_size:]
                    except Exception as e2:
                        print(f"Generation failed: {e2}")
                        # Fallback: use the model's forward pass instead of generate
                        outputs = self.model(batch_x)
                        score = outputs.logits if hasattr(outputs, 'logits') else outputs
                        # Take last win_size tokens
                        score = score[:, -self.win_size:] if score.shape[1] >= self.win_size else score

                score = score.detach().cpu().numpy()
                test_scores.append(score)
                test_labels.append(batch_y)

        test_scores = np.concatenate(test_scores, axis=0).reshape(-1, 1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1, 1)

        print("Test scores shape:", test_scores.shape)
        print("Test labels shape:", test_labels.shape)

        return test_scores.reshape(-1)