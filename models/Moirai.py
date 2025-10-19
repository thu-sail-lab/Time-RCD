"""
Moirai model for anomaly detection using zero-shot forecasting.
Adapted from test_anomaly.py approach for TSB-AD framework.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

from .base import BaseDetector
from ..utils.dataset import MoiraiWindowedDataset


class Moirai(BaseDetector):
    def __init__(self, 
                 win_size=96,
                 model_path="Salesforce/moirai-1.0-R-small",
                 num_samples=100,
                 device='cuda:0',
                 use_score=False,
                 threshold=0.5):
        """
        Initialize Moirai anomaly detector.
        
        Args:
            win_size (int): Window size for context and prediction
            model_path (str): Path to pretrained Moirai model
            num_samples (int): Number of forecast samples
            device (str): Device to run model on
            use_score (bool): Whether to use raw scores or threshold
            threshold (float): Threshold for binary classification
        """
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
        Fit the Moirai model and compute anomaly scores.
        
        Args:
            data: Input time series data (1D or 2D numpy array)
        """
        try:
            # Ensure data is in the right format
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            print(f"Moirai: Processing data with shape {data.shape}")
            
            # Create windowed dataset following test_anomaly.py pattern
            dataset = MoiraiWindowedDataset(
                data=data, 
                win_size=self.win_size, 
                step=self.win_size,  # Non-overlapping windows
                normalize=False  # Let Moirai handle normalization
            )
            
            print(f"Moirai: Created {len(dataset)} windows")
            
            if len(dataset) == 0:
                print("Warning: No valid windows created. Data might be too short.")
                self.decision_scores_ = np.zeros(len(data))
                return
            
            # Process each window using DataLoader (similar to test_anomaly.py)
            data_loader = DataLoader(
                dataset=dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False
            )
            
            all_predictions = []
            all_targets = []
            
            print("Processing windows with Moirai model...")
            # Add progress bar for window processing
            for i, (context, target) in enumerate(tqdm(data_loader, desc="Processing windows", unit="window")):
                # Process single window following test_anomaly.py pattern
                scores = self._process_window(context.squeeze(0).numpy(), target.squeeze(0).numpy(), i)
                all_predictions.append(scores)
                all_targets.append(target.squeeze(0).numpy())
            
            # Combine all predictions
            if all_predictions:
                print("Computing anomaly scores...")
                # Concatenate predictions along time dimension
                combined_predictions = np.concatenate(all_predictions, axis=0)
                combined_targets = np.concatenate(all_targets, axis=0)
                
                # Compute anomaly scores as prediction error
                if combined_targets.ndim == 1 or combined_predictions.ndim == 1:
                    # Handle univariate case or when predictions are 1D
                    if combined_targets.ndim != combined_predictions.ndim:
                        # Ensure both have same number of dimensions
                        if combined_predictions.ndim == 1 and combined_targets.ndim == 2:
                            combined_predictions = combined_predictions.reshape(-1, 1)
                        elif combined_targets.ndim == 1 and combined_predictions.ndim == 2:
                            combined_targets = combined_targets.reshape(-1, 1)
                    
                    if combined_targets.shape != combined_predictions.shape:
                        print(f"Shape mismatch: targets {combined_targets.shape}, predictions {combined_predictions.shape}")
                        # Use only the first feature if shapes don't match
                        if combined_targets.ndim == 2:
                            combined_targets = combined_targets[:, 0]
                        if combined_predictions.ndim == 2:
                            combined_predictions = combined_predictions[:, 0]
                    
                    anomaly_scores = (combined_targets - combined_predictions) ** 2
                    if anomaly_scores.ndim == 2:
                        anomaly_scores = np.mean(anomaly_scores, axis=1)
                else:
                    # For multivariate, use mean squared error across features
                    if combined_targets.shape != combined_predictions.shape:
                        print(f"Shape mismatch: targets {combined_targets.shape}, predictions {combined_predictions.shape}")
                        # Use only matching dimensions
                        min_features = min(combined_targets.shape[1], combined_predictions.shape[1])
                        combined_targets = combined_targets[:, :min_features]
                        combined_predictions = combined_predictions[:, :min_features]
                    
                    anomaly_scores = np.mean((combined_targets - combined_predictions) ** 2, axis=1)
                
                # Pad scores to match original data length
                print("Padding scores to original data length...")
                self.decision_scores_ = self._pad_scores_to_original_length(
                    anomaly_scores, len(data), dataset.get_window_info()
                )
            else:
                print("Warning: No predictions generated")
                self.decision_scores_ = np.zeros(len(data))
                
        except Exception as e:
            print(f"Error in Moirai.fit(): {str(e)}")
            import traceback
            traceback.print_exc()
            self.decision_scores_ = np.zeros(len(data))

    def _process_window(self, context, target, window_index):
        """
        Process a single window following the test_anomaly.py approach.
        
        Args:
            context: Context data for the window (win_size, n_features)
            target: Target data for the window (win_size, n_features) 
            window_index: Index of the current window
            
        Returns:
            predictions: Forecasted values for the target period
        """
        try:
            # Update progress description in tqdm (this will be shown in the progress bar)
            tqdm.write(f"Processing window {window_index + 1}")
            
            # Ensure 2D shape
            if context.ndim == 1:
                context = context.reshape(-1, 1)
            if target.ndim == 1:
                target = target.reshape(-1, 1)
            
            # Combine context and target for full window (following test_anomaly.py)
            full_window = np.vstack([context, target])
            
            # Create DataFrame
            feature_df = pd.DataFrame(full_window)
            
            # For multivariate data, we need to handle it properly
            if feature_df.shape[1] == 1:
                feature_df.columns = ['target']
                target_col = 'target'
                feature_cols = []
            else:
                # For multivariate, use all features as target
                feature_df.columns = [f'target_{i}' for i in range(feature_df.shape[1])]
                target_col = feature_df.columns.tolist()  # Use all columns as targets
                feature_cols = []
            
            # Add timestamp and unique_id
            timestamp_range = pd.date_range(
                start=pd.Timestamp('2023-01-01 10:00:00'), 
                periods=len(feature_df), 
                freq='T'
            )
            feature_df.index = timestamp_range
            feature_df['unique_id'] = window_index
            
            # Create GluonTS dataset
            moirai_df = feature_df.reset_index().rename(columns={'index': 'timestamp'})
            
            if isinstance(target_col, list):
                # Multivariate case - use multiple target columns
                ds = PandasDataset.from_long_dataframe(
                    moirai_df,
                    target=target_col,
                    item_id="unique_id",
                    timestamp="timestamp",
                )
            else:
                # Univariate case
                if feature_cols:
                    ds = PandasDataset.from_long_dataframe(
                        moirai_df,
                        target=target_col,
                        item_id="unique_id",
                        timestamp="timestamp",
                        feat_dynamic_real=feature_cols,
                    )
                else:
                    ds = PandasDataset.from_long_dataframe(
                        moirai_df,
                        target=target_col,
                        item_id="unique_id",
                        timestamp="timestamp",
                    )
            
            # Split dataset (following test_anomaly.py)
            test_size = self.win_size
            _, test_template = split(ds, offset=-test_size)
            
            test_data = test_template.generate_instances(
                prediction_length=self.win_size,
                windows=1,
                distance=self.win_size,
                max_history=self.win_size,
            )
            
            # Create Moirai model
            # Determine target dimension based on number of features
            target_dim = target.shape[1] if target.ndim > 1 else 1
            
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(self.model_path),
                prediction_length=self.win_size,
                context_length=self.win_size,
                patch_size="auto",
                num_samples=self.num_samples,
                target_dim=target_dim,
                feat_dynamic_real_dim=ds.num_feat_dynamic_real,
                past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
            )
            
            # Create predictor and generate forecasts
            predictor = model.create_predictor(batch_size=1, device=self.device)
            forecasts = predictor.predict(test_data.input)
            forecasts = list(forecasts)
            
            # Get median prediction (following test_anomaly.py)
            predictions = np.median(forecasts[0].samples, axis=0)
            
            return predictions
            
        except Exception as e:
            print(f"Error processing window {window_index}: {str(e)}")
            # Return zeros as fallback with correct shape
            target_shape = (self.win_size, target.shape[1]) if target.ndim > 1 else (self.win_size,)
            return np.zeros(target_shape)

    def _pad_scores_to_original_length(self, scores, original_length, window_info):
        """
        Pad anomaly scores to match the original data length.
        
        Args:
            scores: Computed anomaly scores from windows
            original_length: Length of the original input data
            window_info: Information about windowing strategy
            
        Returns:
            padded_scores: Scores padded to original length
        """
        padded_scores = np.zeros(original_length)
        
        win_size = window_info['win_size']
        step = window_info['step']
        
        # Fill in scores from each window
        score_windows = scores.reshape(-1, win_size)
        for i, score_window in enumerate(tqdm(score_windows, desc="Padding scores", unit="window")):
            start_idx = i * step + win_size  # Offset by win_size (context part)
            end_idx = start_idx + win_size
            
            if end_idx <= original_length:
                padded_scores[start_idx:end_idx] = score_window
            elif start_idx < original_length:
                # Partial window at the end
                remaining = original_length - start_idx
                padded_scores[start_idx:] = score_window[:remaining]
        
        # Fill beginning (context part) with first window's average
        if len(scores) > 0:
            first_score = np.mean(scores[:win_size]) if len(scores) >= win_size else np.mean(scores)
            padded_scores[:win_size] = first_score
        
        return padded_scores

    def decision_function(self, X):
        """
        Not used for zero-shot approach, present for API consistency.
        """
        return self.decision_scores_
