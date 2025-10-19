# Time-RCD: Time Series Anomaly Detection with TSB-AD Benchmark

This repository contains the implementation of Time-RCD for time series anomaly detection, integrated with the TSB-AD (Time Series Benchmark for Anomaly Detection) datasets.

## Prerequisites

- Python 3.10
- conda (recommended for environment management)
- Git

## Installation

### 1. Create and Activate Conda Environment

```bash
conda create -n Time-RCD python=3.10
conda activate Time-RCD
```

### 2. Clone the Repository

```bash
git clone https://github.com/thu-sail-lab/Time-RCD.git
cd Time-RCD
```

### 3. Download TSB-AD Datasets

Create the datasets directory and download the TSB-AD-U (univariate) and TSB-AD-M (multivariate) datasets:

```bash
mkdir -p "datasets" \
  && wget -O "datasets/TSB-AD-U.zip" "https://www.thedatum.org/datasets/TSB-AD-U.zip" \
  && wget -O "datasets/TSB-AD-M.zip" "https://www.thedatum.org/datasets/TSB-AD-M.zip" \
  && cd datasets \
  && unzip TSB-AD-U.zip && rm TSB-AD-U.zip \
  && unzip TSB-AD-M.zip && rm TSB-AD-M.zip \
  && cd ..
```

### 4. Install Python Dependencies

#### Option A: Fast Install (using uv)

```bash
pip install uv
uv pip install jaxtyping einops pandas numpy scikit-learn transformers torch torchvision statsmodels matplotlib seaborn -U "huggingface_hub[cli]"
```

#### Option B: Normal Install

```bash
pip install jaxtyping einops pandas numpy scikit-learn transformers torch torchvision statsmodels matplotlib seaborn -U "huggingface_hub[cli]"
```

### 5. Download Pre-trained Checkpoints

Download the pre-trained model checkpoints from Hugging Face:

```bash
huggingface-cli download thu-sail-lab/Time_RCD checkpoints.zip
unzip checkpoints.zip
cd ..
```

## Usage

⚠️ **Important**: You must run the commands from the parent directory of the Time-RCD repository to avoid relative import problems. Make sure you have executed `cd ..` after downloading the checkpoints.

### Single Variable Time Series

To run anomaly detection on univariate time series:

```bash
python -m TimeRCD.main
```

### Multi-Variable Time Series

To run anomaly detection on multivariate time series:

```bash
python -m TimeRCD.main mode multi
```

**Note**: If you run the scripts from within the Time-RCD directory or any other location, you will encounter import errors due to relative import issues. Always run from the parent directory using the module syntax `python -m TimeRCD.main`.

## Project Structure

```
.
├── checkpoints/          # Pre-trained model checkpoints
├── datasets/            # TSB-AD datasets (univariate and multivariate)
├── evaluation/          # Evaluation metrics and visualization tools
├── models/              # Model implementations
│   └── time_rcd/       # Time-RCD model components
├── utils/               # Utility functions
├── main.py              # Main entry point
├── model_wrapper.py     # Model wrapper for different algorithms
└── README.md            # This file
```

## Citation

If you use this code or the Time-RCD model in your research, please cite:

```bibtex
@article{time-rcd,
  title={Time-RCD: Time Series Anomaly Detection with Reconstruction-based Contrastive Learning},
  author={Authors},
  journal={Journal},
  year={2024}
}
```

## License

[Specify your license here]

## Acknowledgments

- TSB-AD Dataset: [The Datum](https://www.thedatum.org/)
- Time-RCD Model: [thu-sail-lab](https://github.com/thu-sail-lab)

