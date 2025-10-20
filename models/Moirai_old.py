"""
Moirai model wrapper for anomaly detection
Adapted from the test_anomaly.py implementation
"""

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import tempfile
import warnings
warnings.filterwarnings('ignore')

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai.forecast import MoiraiForecast, MoiraiModule

from .base import BaseDetector


class Moirai(BaseDetector):
    def __init__(self, 
                 win_size=96,
                 model_path="Salesforce/moirai-1.0-R-small",
                 num_samples=100,
                 device='cuda:0',
                 use_score=False,
                 threshold=0.5):

        self.model_name = 'Moirai'
        self.win_size = win_size
        self.model_path = model_path
        self.num_samples = num_samples
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.use_score = use_score
        self.threshold = threshold
        self.decision_scores_ = None

    def fit(self, data):
        """
        Fit Moirai on the data and compute anomaly scores using zero-shot approach
        This implementation follows the exact windowing logic from the data loaders
        """
        print(f"Moirai zero-shot anomaly detection on data shape: {data.shape}")
        
        # Handle univariate data (ensure 2D shape)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Check if we have enough data
        if data.shape[0] < 2 * self.win_size:
            raise ValueError(f"Data length ({data.shape[0]}) is less than required minimum (2 * win_size = {2 * self.win_size})")
        
        all_target = []
        all_moirai_preds = []
        last_pred_label = None
        
        # Create sliding windows following the data loader pattern
        # For testing, we use stride = win_size (non-overlapping windows like in data loaders)
        num_windows = (data.shape[0] - 2 * self.win_size) // self.win_size + 1
        
        for i in tqdm(range(num_windows), desc="Processing windows"):
            # Extract window following data loader logic
            start_idx = i * self.win_size
            end_idx = start_idx + 2 * self.win_size
            
            if end_idx > data.shape[0]:
                break
                
            # Get the 2*win_size window (this matches batch_x from data loader)
            window_data = data[start_idx:end_idx]  # Shape: (2*win_size, n_features)
            
            # Create synthetic labels (all zeros initially, replaced by predictions)
            label = np.zeros(window_data.shape[0])
            
            # Replace the first win_size labels with last prediction if not first window
            if i != 0 and last_pred_label is not None:
                label[:self.win_size] = last_pred_label
            
            # Convert to DataFrame format required by GluonTS
            # Handle both univariate and multivariate data
            if window_data.shape[1] == 1:
                # Univariate case
                feature = pd.DataFrame(window_data, columns=['value'])
            else:
                # Multivariate case
                feature = pd.DataFrame(window_data)
                feature.columns = [f'feature_{j}' for j in range(feature.shape[1])]
            
            label_df = pd.DataFrame(label, columns=['label'])
            df = pd.concat([feature, label_df], axis=1)
            
            # Add timestamp and unique_id
            new_index = pd.date_range(
                start=pd.Timestamp('2023-01-01 10:00:00'), 
                periods=len(df), 
                freq='T'
            )
            new_index_iso = new_index.strftime('%Y-%m-%d %H:%M:%S')
            df.insert(0, 'Timestamp', new_index_iso)
            df['unique_id'] = 0
            moirai_df = df.set_index('Timestamp')

            # Create GluonTS dataset
            feat_cols = feature.columns.tolist()
            ds = PandasDataset.from_long_dataframe(
                moirai_df,
                target="label",
                item_id="unique_id",
                feat_dynamic_real=feat_cols,
            )

            test_size = self.win_size
            _, test_template = split(ds, offset=-test_size)

            test_data = test_template.generate_instances(
                prediction_length=self.win_size,
                windows=1,
                distance=self.win_size,
                max_history=self.win_size,
            )

            # Create Moirai model (recreate for each window to avoid memory issues)
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(self.model_path),
                prediction_length=self.win_size,
                context_length=self.win_size,
                patch_size="auto",
                num_samples=self.num_samples,
                target_dim=1,
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            )

            try:
                predictor = model.create_predictor(batch_size=1, device=self.device)
                forecasts = predictor.predict(test_data.input)
                forecasts = list(forecasts)

                moirai_preds = np.median(forecasts[0].samples, axis=0)
                all_moirai_preds.append(moirai_preds)
                
                # Collect targets for verification
                input_it = iter(test_data.label)
                for item in input_it:
                    all_target.extend(item['target'])
                
                # Update last prediction for next window
                if self.use_score:
                    last_pred_label = moirai_preds
                else:
                    last_pred_label = (moirai_preds >= self.threshold).astype(int)
                    
            except Exception as e:
                print(f"Error processing window {i}: {e}")
                # Use zeros as fallback
                moirai_preds = np.zeros(self.win_size)
                all_moirai_preds.append(moirai_preds)
                last_pred_label = moirai_preds

        # Concatenate all predictions
        if all_moirai_preds:
            all_moirai_preds = np.concatenate(all_moirai_preds, axis=0)
        else:
            all_moirai_preds = np.zeros(0)
        
        # Create scores array that matches the original data length
        # This follows the pattern from data loaders: each window predicts win_size points
        padded_scores = np.zeros(data.shape[0])
        
        if len(all_moirai_preds) > 0:
            # Map predictions back to original data indices
            for i, pred_window in enumerate(np.array_split(all_moirai_preds, num_windows)):
                if len(pred_window) > 0:
                    start_pred_idx = self.win_size + i * self.win_size  # Start from win_size offset
                    end_pred_idx = min(start_pred_idx + len(pred_window), data.shape[0])
                    actual_len = end_pred_idx - start_pred_idx
                    padded_scores[start_pred_idx:end_pred_idx] = pred_window[:actual_len]
            
            # Fill the first win_size points with the first prediction if available
            if self.win_size < len(padded_scores):
                first_pred = all_moirai_preds[0] if len(all_moirai_preds) > 0 else 0
                padded_scores[:self.win_size] = first_pred
        
        self.decision_scores_ = padded_scores
        print(f"Generated anomaly scores shape: {self.decision_scores_.shape}")
        return self

    def decision_function(self, X):
        """
        Return anomaly scores for X
        """
        if self.decision_scores_ is None:
            raise ValueError("Model must be fitted before calling decision_function")
        return self.decision_scores_[:len(X)]

    def zero_shot(self, data):
        """
        Zero-shot anomaly detection
        """
        self.fit(data)
        return self.decision_scores_
