# Time-RCD Tutorial

Time-RCD is a zero-shot foundation model for time series anomaly detection. Given a univariate or multivariate series, it outputs an anomaly score in `[0, 1]` for each time step. Higher scores indicate a higher likelihood of an anomaly.

## 1. Installation

Python 3.10 is recommended:

```bash
conda create -n Time-RCD python=3.10
conda activate Time-RCD

git clone https://github.com/thu-sail-lab/Time-RCD.git
cd Time-RCD
pip install .
```

If you are in mainland China, set the Hugging Face mirror before the first automatic checkpoint download:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

This variable only affects Hugging Face downloads. Cached checkpoints and weights loaded from local paths are unaffected.

## 2. Univariate data

Univariate input can be a NumPy array of shape `(T,)` or `(T, 1)`, where `T` is the number of time steps.

```python
import numpy as np

from time_rcd import TimeRCDDetector

data = np.load("series.npy")  # shape: (T,) or (T, 1)

detector = TimeRCDDetector.from_pretrained(variant="uni")
scores = detector.predict(data)

print(scores.shape)  # (T,)
```

## 3. Multivariate data

Multivariate input must have shape `(T, C)` with `C > 1`. Each column is one feature or sensor channel.

```python
import numpy as np

from time_rcd import TimeRCDDetector

data = np.load("multivariate_series.npy")  # shape: (T, C), C > 1

detector = TimeRCDDetector.from_pretrained(variant="multi")
scores = detector.predict(data)

print(scores.shape)  # (T,)
```

The model is initialized on the first call to `predict()`, using the channel count `C`. If the same detector is later used on data with a different number of channels, the model is rebuilt automatically.

## 4. Loading from CSV

Rows should be in chronological order. Timestamp columns are not model features and should be removed before inference.

```python
import pandas as pd

from time_rcd import TimeRCDDetector

frame = pd.read_csv("sensor_data.csv")
feature_columns = ["temperature", "pressure", "flow"]
data = frame[feature_columns].to_numpy()

detector = TimeRCDDetector.from_pretrained(variant="multi")
scores = detector.predict(data)

frame["anomaly_score"] = scores
frame.to_csv("scored_sensor_data.csv", index=False)
```

For a univariate CSV, select one numeric column and use `variant="uni"`:

```python
data = frame["value"].to_numpy()
detector = TimeRCDDetector.from_pretrained(variant="uni")
scores = detector.predict(data)
```

## 5. Local checkpoints and offline inference

If checkpoints are already on disk, you can avoid network access:

```python
from time_rcd import TimeRCDDetector

detector = TimeRCDDetector.from_local(
    "best_model/pretrain_checkpoint_best_uni.pth",
    variant="uni",
)
scores = detector.predict(data)
```

To use the local cache without contacting the Hub:

```python
detector = TimeRCDDetector.from_pretrained(
    variant="uni",
    local_files_only=True,
)
```

If nothing is cached, `local_files_only=True` will fail. Download the checkpoint once while online, or use `from_local()` instead.

## 6. Common parameters

```python
detector = TimeRCDDetector.from_pretrained(
    variant="uni",
    win_size=5000,
    batch_size=64,
    device="cuda",  # or "cpu"
)
```

- `win_size`: Sliding window length. Default is `5000`. Shorter sequences use the full sequence length.
- `batch_size`: Inference batch size. Defaults to `64` (uni) or `1` (multi). Reduce it if you run out of GPU memory.
- `device`: When omitted, CUDA is used if available; otherwise CPU.
- `return_logits=True`: Return raw anomaly logits in addition to probability scores.

The same detector can be reused on sequences of different lengths. A short sequence does not change the window setting for later predictions.

## 7. Interpreting anomaly scores

`scores` has one value per input time step, in `[0, 1]`. Higher values mean the model considers that time step more anomalous. Time-RCD does not ship a universal fixed threshold, because a good threshold depends on your data distribution, false-alarm cost, and expected anomaly rate.

A simple starting point is a high quantile of scores from known normal data:

```python
import numpy as np

threshold = np.quantile(normal_scores, 0.995)
is_anomaly = scores >= threshold
```

Calibrate the threshold on labeled validation data or known normal history. Do not deploy the example quantile directly to production without evaluation.
