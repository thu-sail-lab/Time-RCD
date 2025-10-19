import json
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os
import pickle
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path


class ChatTSAnomalyPretrainDataset(Dataset):
    def __init__(self,
                 dataset_dir: str,
                 filename: str,
                 split: str = 'train',
                 train_ratio: float = 0.95,
                 seed: int = 42):
        file_path = os.path.join(dataset_dir, filename)
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        random.seed(seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        num_train = int(len(dataset) * train_ratio)
        if split == 'train':
            selected_indices = indices[:num_train]
        elif split == 'test':
            selected_indices = indices[num_train:]
        else:
            raise ValueError("split must be 'train' or 'test'")
        self.data = [dataset[i] for i in selected_indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        time_series = torch.tensor(sample['time_series'], dtype=torch.float32)
        normal_time_series = torch.tensor(sample['normal_time_series'], dtype=torch.float32)
        labels = torch.tensor(sample['labels'], dtype=torch.long)
        attribute = sample['attribute']
        return time_series, normal_time_series, labels, attribute


class ChatTSAnomalyQADataset(Dataset):
    """Dataset class for time series anomaly detection with QA pairs.

    This dataset loads time series data and corresponding question-answer pairs
    for anomaly detection tasks. It supports train/val split and efficient loading
    of series data from the anomaly_llava_datasets format.

    Attributes:
        split (str): Dataset split, either 'train' or 'val'
        series_dir (Path): Directory containing series JSON files
        metadata (Dict): Dataset metadata loaded from metadata.json
        series_files (List[str]): List of series file paths
        window_size_range (Tuple[int, int]): Range of window sizes used in the dataset
    """

    def __init__(
            self,
            dataset_dir: str,
            split: str = 'train',
            train_ratio: float = 0.95,
            seed: int = 42,
            cache_size: int = 1000
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset_dir: Path to the dataset directory containing metadata.json and series/
            split: Dataset split, either 'train' or 'val'
            train_ratio: Ratio of training samples (default: 0.8)
            seed: Random seed for reproducibility (default: 42)
            cache_size: Number of series files to keep in memory (default: 1000)
        """
        self.split = split
        self.series_dir = Path(dataset_dir) / 'series'

        # Get all series files and shuffle them
        self.series_files = sorted(self.series_dir.glob('series_*.json'))
        random.seed(seed)
        random.shuffle(self.series_files)

        # Split into train/val
        split_idx = int(len(self.series_files) * train_ratio)
        self.series_files = self.series_files[:split_idx] if split == 'train' else self.series_files[split_idx:]

        # Initialize LRU cache for series data
        self._cache = {}
        self._cache_size = cache_size
        self._cache_order = []

    def _load_series(self, file_path: Path) -> Dict:
        """Load a series file with caching.

        Args:
            file_path: Path to the series JSON file

        Returns:
            Dictionary containing the series data
        """
        if file_path in self._cache:
            # Update cache order
            self._cache_order.remove(file_path)
            self._cache_order.append(file_path)
            return self._cache[file_path]

        # Load new file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Update cache
        if len(self._cache) >= self._cache_size:
            # Remove oldest item
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        self._cache[file_path] = data
        self._cache_order.append(file_path)
        return data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.series_files)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, List[Dict]]]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing:
                - time_series: Time series data as torch.Tensor
                - windows: List of window data containing QA pairs
                - sample_id: Unique identifier for the sample
        """
        file_path = self.series_files[idx]
        data = self._load_series(file_path)

        # Convert time series to tensor
        time_series = np.array(data['original_data']['time_series'])
        time_series_tensor = torch.FloatTensor(time_series)

        return {
            'time_series': time_series_tensor,
            'analysis_data': data['windows']
        }

