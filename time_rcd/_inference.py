"""Minimal inference backend packaged with the public Time-RCD API."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ._core.TimeRCD_pretrain_multi import TimeSeriesPretrainModel
from ._core.time_rcd_config import TimeRCDConfig


class _WindowDataset(Dataset):
    """Split a normalized time series into non-overlapping padded windows."""

    def __init__(self, data: np.ndarray, window_size: int) -> None:
        mean = np.mean(data, axis=0)
        std = np.where(np.std(data, axis=0) == 0, 1e-8, np.std(data, axis=0))
        normalized = (data - mean) / std

        padding = (-len(normalized)) % window_size
        if padding:
            normalized = np.vstack(
                [normalized, np.repeat(normalized[-1:, :], padding, axis=0)]
            )
        self.data = normalized
        self.window_size = window_size
        self.original_length = len(data)

    def __len__(self) -> int:
        return len(self.data) // self.window_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = index * self.window_size
        end = start + self.window_size
        valid_length = min(self.window_size, self.original_length - start)
        mask = torch.zeros(self.window_size, dtype=torch.bool)
        mask[:valid_length] = True
        return (
            torch.tensor(self.data[start:end], dtype=torch.float32),
            mask,
        )


def _collate_windows(
    batch: list[Tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    time_series, attention_mask = zip(*batch)
    return {
        "time_series": torch.stack(time_series),
        "attention_mask": torch.stack(attention_mask),
    }


class TimeRCDPretrainTester:
    """Inference-only wrapper for a pretrained Time-RCD checkpoint."""

    def __init__(self, checkpoint_path: str, config: TimeRCDConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.win_size = config.win_size
        self.batch_size = config.batch_size
        self.model = TimeSeriesPretrainModel(config).to(self.device)
        self.load_checkpoint(checkpoint_path)
        self.model.eval()

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        self.model.load_state_dict(state_dict)

    def zero_shot(
        self, data: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        window_size = min(len(data), self.win_size)
        dataset = _WindowDataset(data, window_size)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=_collate_windows,
            num_workers=0,
            shuffle=False,
        )

        scores: list[np.ndarray] = []
        logits: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                time_series = batch["time_series"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                local_embeddings = self.model(
                    time_series=time_series, mask=attention_mask
                )
                anomaly_logits = self.model.anomaly_head(local_embeddings)
                anomaly_logits = torch.mean(anomaly_logits, dim=-2)
                anomaly_probs = F.softmax(anomaly_logits, dim=-1)[..., 1]
                scores.append(anomaly_probs.cpu().numpy())
                logits.append(
                    (anomaly_logits[..., 1] - anomaly_logits[..., 0]).cpu().numpy()
                )
        return scores, logits
