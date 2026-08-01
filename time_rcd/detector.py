"""User-facing zero-shot inference API for Time-RCD."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import numpy as np

HF_REPO_ID = "thu-sail-lab/Time-RCD"
CHECKPOINT_FILES = {
    "uni": "best_model/pretrain_checkpoint_best_uni.pth",
    "multi": "best_model/pretrain_checkpoint_best_multi.pth",
}
DEFAULT_WIN_SIZE = 5000
DEFAULT_BATCH_SIZE = {"uni": 64, "multi": 1}


class TimeRCDDetector:
    """Zero-shot time series anomaly detector powered by Time-RCD.

    Parameters
    ----------
    checkpoint_path:
        Path to a local ``.pth`` checkpoint.
    variant:
        ``"uni"`` for univariate series, ``"multi"`` for multivariate series.
    win_size:
        Sliding window length. Sequences shorter than this value use the full
        sequence length instead.
    batch_size:
        Inference batch size. Defaults to 64 (uni) or 1 (multi).
    device:
        PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``. Auto-detected when
        omitted.

    Notes
    -----
    The model is initialized on the first call to :meth:`predict`, when the
    number of input features is known.
    """

    def __init__(
        self,
        checkpoint_path: str,
        variant: Literal["uni", "multi"] = "uni",
        win_size: int = DEFAULT_WIN_SIZE,
        batch_size: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        if variant not in CHECKPOINT_FILES:
            raise ValueError(f"variant must be one of {list(CHECKPOINT_FILES)}, got {variant!r}")

        self.variant = variant
        self.win_size = win_size
        self.batch_size = batch_size if batch_size is not None else DEFAULT_BATCH_SIZE[variant]
        self.checkpoint_path = str(checkpoint_path)
        self.device = device
        self._tester = None
        self._num_features: Optional[int] = None

    def _ensure_tester(self, num_features: int) -> None:
        """Initialize a model compatible with ``num_features`` when needed."""
        if self._tester is not None and self._num_features == num_features:
            return

        from ._core.time_rcd_config import default_config
        from ._inference import TimeRCDPretrainTester

        # ``default_config`` is a module-level template. Each detector must own
        # its configuration so its runtime options cannot affect other instances.
        config = deepcopy(default_config)
        config.ts_config.patch_size = 16
        config.win_size = self.win_size
        config.batch_size = self.batch_size
        config.ts_config.num_features = num_features

        self._tester = TimeRCDPretrainTester(self.checkpoint_path, config)
        self._num_features = num_features
        if self.device is not None:
            import torch

            self._tester.device = torch.device(self.device)
            self._tester.model.to(self._tester.device)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = HF_REPO_ID,
        variant: Literal["uni", "multi"] = "uni",
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> "TimeRCDDetector":
        """Load a checkpoint from Hugging Face Hub (cached locally after first use)."""
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=CHECKPOINT_FILES[variant],
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        return cls(checkpoint_path=checkpoint_path, variant=variant, **kwargs)

    @classmethod
    def from_local(
        cls,
        checkpoint_path: Union[str, Path],
        variant: Literal["uni", "multi"] = "uni",
        **kwargs,
    ) -> "TimeRCDDetector":
        """Load a checkpoint from a local path."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return cls(checkpoint_path=str(path), variant=variant, **kwargs)

    def predict(
        self,
        data: Union[np.ndarray, list],
        return_logits: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Run zero-shot anomaly scoring on a single time series.

        Parameters
        ----------
        data:
            Array of shape ``(T,)`` or ``(T, C)`` where ``T`` is time steps and
            ``C`` is the number of channels.
        return_logits:
            When ``True``, also return raw anomaly logits.

        Returns
        -------
        scores:
            Anomaly scores in ``[0, 1]``, one value per time step.
        logits:
            Returned only when ``return_logits=True``.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim != 2:
            raise ValueError(f"data must be 1D or 2D, got shape {data.shape}")

        if self.variant == "uni" and data.shape[1] != 1:
            raise ValueError(
                "variant='uni' expects a univariate series with shape (T,) or (T, 1). "
                f"Got {data.shape[1]} channels; use variant='multi' instead."
            )
        if self.variant == "multi" and data.shape[1] < 2:
            raise ValueError(
                "variant='multi' expects at least two channels with shape (T, C), where C > 1. "
                "Use variant='uni' for a univariate series."
            )

        original_length = data.shape[0]
        self._ensure_tester(data.shape[1])
        assert self._tester is not None

        score_chunks, logit_chunks = self._tester.zero_shot(data)
        scores = np.concatenate([np.asarray(chunk).reshape(-1) for chunk in score_chunks], axis=0)
        logits = np.concatenate([np.asarray(chunk).reshape(-1) for chunk in logit_chunks], axis=0)

        scores = scores[:original_length]
        logits = logits[:original_length]

        if return_logits:
            return scores, logits
        return scores
