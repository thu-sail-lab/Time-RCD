<div align="center">

# Time-RCD

_Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy_

[![arXiv](https://img.shields.io/badge/arXiv-2509.21190-b31b1b.svg)](https://arxiv.org/abs/2509.21190)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Demo-yellow)](https://huggingface.co/spaces/thu-sail-lab/Time_RCD)
[![时空探索之旅](https://img.shields.io/badge/时空探索之旅-black?logo=wechat&logoColor=white)](https://mp.weixin.qq.com/s/79M3jsEhMKBzbNYpROOBCw)

</div>

<p align="center">
    📰&nbsp;<a href="#-news">News</a>
    | 🔍&nbsp;<a href="#-about">About</a>
    | 🎯&nbsp;<a href="#-use-on-your-own-data">Use on Your Own Data</a>
    | 📁&nbsp;<a href="#-project-structure">Project Structure</a>
    | 🔗&nbsp;<a href="#-citation">Citation</a>
</p>

## 📰 News

- **2026.05:** Time-RCD has been accepted by **ICML 2026**. We also release the [pre-trained dataset generation code and hyperparameters](https://github.com/thu-sail-lab/TSAD_dataset_gen_public).

- **2026.04:** With a new dataset and new checkpoints, Time-RCD achieves better results. The univariate setting improves VUS-PR by an **absolute 6.7 points**, and the multivariate setting improves VUS-PR by an **absolute 4.5 points**.

## 🔍 About

**Time-RCD** is a zero-shot foundation model for time series anomaly detection. Given a univariate or multivariate series, it outputs a per-timestep anomaly score without any task-specific training on your data.

🐘 On the [TSB-AD benchmark](https://thedatumorg.github.io/TSB-AD/), Time-RCD achieves a **Univariate VUS-PR of 0.52** and a **Multivariate VUS-PR of 0.32**.

**[🌟 Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/thu-sail-lab/Time_RCD)** — try Time-RCD interactively in your browser.

<div align="center">
<img src="https://raw.githubusercontent.com/thu-sail-lab/Time-RCD/main/zero-shot.png" style="width:95%;" />
</div>

This repository contains:

1. **`time_rcd/`** — a lightweight Python API for inference on your own data

For a step-by-step guide, see **[Tutorial.md](https://github.com/thu-sail-lab/Time-RCD/blob/main/Tutorial.md)**.

---

## 🎯 Use on Your Own Data

### Installation

```bash
conda create -n Time-RCD python=3.10
conda activate Time-RCD

git clone https://github.com/thu-sail-lab/Time-RCD.git
cd Time-RCD
pip install .
```

### Python API (recommended)

Checkpoints are downloaded from Hugging Face automatically on first use and cached locally.
For servers in China, set `HF_ENDPOINT=https://hf-mirror.com` before running
the examples or loading a checkpoint.

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

```python
import numpy as np
from time_rcd import TimeRCDDetector

data = np.load("my_series.npy")  # shape (T,) or (T, C)

detector = TimeRCDDetector.from_pretrained(variant="uni")   # or "multi"
scores = detector.predict(data)                             # shape (T,)
```

**Multivariate series** — use `variant="multi"` when `C > 1`:

```python
detector = TimeRCDDetector.from_pretrained(variant="multi")
scores = detector.predict(multivariate_data)  # shape (T, C) -> scores (T,)
```

**Local checkpoint** — if you already downloaded weights:

```python
detector = TimeRCDDetector.from_local(
    "best_model/pretrain_checkpoint_best_uni.pth",
    variant="uni",
)
```

### Quick example

```bash
python examples/quickstart.py
```

See **[Tutorial.md](https://github.com/thu-sail-lab/Time-RCD/blob/main/Tutorial.md)** for CSV loading, hyperparameters, and more examples.

---

## 📁 Project Structure

```
.
├── time_rcd/              # User-facing inference API
│   ├── detector.py        # TimeRCDDetector
│   └── _core/             # Time-RCD inference model implementation
├── examples/
│   └── quickstart.py      # Minimal inference example
├── Tutorial.md            # Guide for your own data
├── pyproject.toml         # Package metadata and dependencies
├── zero-shot.png          # Model overview
└── README.md
```

### TSB-AD benchmark code

The original benchmark integration, evaluation scripts, and baseline
implementations are maintained in the
[`tsb-ad-integration`](https://github.com/thu-sail-lab/Time-RCD/tree/tsb-ad-integration)
branch. For the lightweight zero-shot inference API, use the `main` branch.

---

## 🔗 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{lan2025foundationmodelszeroshottime,
      title={Towards Foundation Models for Zero-Shot Time Series Anomaly Detection: Leveraging Synthetic Data and Relative Context Discrepancy}, 
      author={Tian Lan and Hao Duong Le and Jinbo Li and Wenjun He and Meng Wang and Chenghao Liu and Chen Zhang},
      year={2025},
      eprint={2509.21190},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2509.21190}, 
}
```
