#!/usr/bin/env python3
"""
Upload Time-RCD Model to HuggingFace Hub

This script prepares and uploads your Time-RCD model to HuggingFace Hub.

Usage:
    1. Install: pip install huggingface_hub
    2. Login: huggingface-cli login
    3. Edit USERNAME below
    4. Run: python upload_to_huggingface.py
"""

from huggingface_hub import HfApi, create_repo, login
import os
import sys
import argparse

# Add huggingface_time_rcd to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'huggingface_time_rcd'))

from huggingface_time_rcd import Time_RCD, TimeRCDProcessor

# ========== CONFIGURATION ==========
# ⚠️ CHANGE THESE VALUES ⚠️
USERNAME = "thu-sail-lab"  # Your HuggingFace username
REPO_NAME = "Time_RCD"      # Repository name
# ==================================

CHECKPOINT_PATH = "/Users/oliver/Documents/2025/Huawei/Time-RCD/Time-RCD/Testing/checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth"
UPLOAD_DIR = "./time_rcd_hf_upload"

# README content
README_TEMPLATE = """---
license: apache-2.0
tags:
  - time-series
  - anomaly-detection
  - zero-shot
  - pytorch
  - transformers
library_name: transformers
pipeline_tag: time-series-classification
---

# Time-RCD: Zero-Shot Time Series Anomaly Detection

Time-RCD is a transformer-based model for zero-shot anomaly detection in time series data.

## Quick Start

```python
from transformers import AutoModel, AutoConfig
import numpy as np

# Load model
model = AutoModel.from_pretrained(
    "{repo_id}",
    trust_remote_code=True
)

# Load processor
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    "{repo_id}",
    trust_remote_code=True
)

# Prepare data
data = np.random.randn(10000, 1)  # [n_samples, n_features]

# Process data
processed = processor(
    data,
    return_tensors="pt"
)

# Get anomaly scores
outputs = model(**processed)
anomaly_scores = outputs.anomaly_scores.numpy()
```

## Model Details

- **Architecture:** Transformer encoder with patch embedding
- **Parameters:** ~5M parameters
- **Patch Size:** 4
- **Hidden Dimension:** 512
- **Projection Dimension:** 256
- **Layers:** 8 transformer layers
- **Attention Heads:** 8 heads

## Features

✅ **Zero-shot detection** - No training required  
✅ **Multi-variate support** - Handle multiple features  
✅ **Flexible windows** - Configurable window sizes  
✅ **Robust normalization** - Built-in preprocessing  

## Usage Examples

### Basic Anomaly Detection

```python
from transformers import AutoModel
import numpy as np

model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)

# Your time series (n_samples, n_features)
data = np.random.randn(10000, 1)

# Get anomaly scores
outputs = model.zero_shot(data)
scores = outputs.anomaly_scores.numpy()

# Detect anomalies (e.g., top 5%)
threshold = np.percentile(scores, 95)
anomalies = scores > threshold
```

### With Custom Processing

```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("{repo_id}", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("{repo_id}", trust_remote_code=True)

# Configure processor
processor.win_size = 5000
processor.normalize = True

# Process and detect
processed = processor(data, return_tensors="pt")
outputs = model(**processed)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| patch_size | 4 | Patch size for embedding |
| d_model | 512 | Model dimension |
| d_proj | 256 | Projection dimension |
| num_layers | 8 | Transformer layers |
| num_heads | 8 | Attention heads |
| use_rope | True | Rotary position embeddings |

## Performance

Evaluated on various time series anomaly detection benchmarks.

## Limitations

- Requires sufficient data (> window size)
- Performance varies by domain
- High-dimensional data may need preprocessing

## Citation

```bibtex
@article{{time-rcd,
  title={{Time-RCD: Zero-Shot Time Series Anomaly Detection}},
  author={{Your Name}},
  year={{2025}}
}}
```

## License

Apache 2.0
"""

def create_readme(repo_id):
    """Create README with proper formatting"""
    return README_TEMPLATE.format(repo_id=repo_id)

def main():
    parser = argparse.ArgumentParser(description='Upload Time-RCD model to HuggingFace')
    parser.add_argument('--yes', '-y', action='store_true', 
                        help='Skip confirmation prompt')
    parser.add_argument('--create-pr', action='store_true',
                        help='Create a pull request instead of pushing directly (use if you lack write access)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Time-RCD HuggingFace Upload Script")
    print("=" * 60)
    
    # Validate username
    if USERNAME == "your-username":
        print("❌ ERROR: Please edit the script and set your HuggingFace username!")
        print("Change USERNAME = 'your-username' to your actual username")
        sys.exit(1)
    
    repo_id = f"{USERNAME}/{REPO_NAME}"
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        sys.exit(1)
    
    print(f"\n📋 Configuration:")
    print(f"   Username: {USERNAME}")
    print(f"   Repository: {REPO_NAME}")
    print(f"   Full repo ID: {repo_id}")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    print(f"   Upload directory: {UPLOAD_DIR}")
    print(f"   Create PR: {'Yes' if args.create_pr else 'No (direct push)'}")
    
    # Confirm
    if not args.yes:
        response = input("\n❓ Continue with upload? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("❌ Upload cancelled")
            sys.exit(0)
    else:
        print("\n✅ Auto-confirmed with --yes flag")
    
    try:
        # Step 1: Login
        print("\n🔐 Step 1: Checking HuggingFace login...")
        try:
            login(token=None, add_to_git_credential=True)
            print("✅ Already logged in")
        except:
            print("Please login to HuggingFace")
            login()
        
        # Step 2: Convert and save model
        print("\n💾 Step 2: Converting checkpoint to HuggingFace format...")
        print(f"   Loading from: {CHECKPOINT_PATH}")
        
        model = Time_RCD.from_original_checkpoint(CHECKPOINT_PATH)
        print("✅ Model loaded successfully")
        
        # Create upload directory
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        print(f"   Saving to: {UPLOAD_DIR}")
        model.save_pretrained(UPLOAD_DIR)
        model.config.save_pretrained(UPLOAD_DIR)
        print("✅ Model saved in HuggingFace format")
        
        # Step 3: Save processor
        print("\n💾 Step 3: Creating processor...")
        processor = TimeRCDProcessor(
            win_size=5000,
            stride=5000,
            normalize=True,
        )
        processor.save_pretrained(UPLOAD_DIR)
        print("✅ Processor saved")
        
        # Step 4: Copy modeling code files
        print("\n📋 Step 4: Copying modeling code files...")
        import shutil
        code_files = [
            ("huggingface_time_rcd/modeling_time_rcd.py", "modeling_time_rcd.py"),
            ("huggingface_time_rcd/configuration_time_rcd.py", "configuration_time_rcd.py"),
            ("huggingface_time_rcd/processing_time_rcd.py", "processing_time_rcd.py"),
        ]
        for local_path, repo_path in code_files:
            if os.path.exists(local_path):
                shutil.copy(local_path, os.path.join(UPLOAD_DIR, repo_path))
                print(f"   ✅ {repo_path}")
            else:
                print(f"   ⚠️  {local_path} not found, skipping")
        
        # Step 5: Create README
        print("\n📝 Step 5: Creating README.md...")
        readme_content = create_readme(repo_id)
        with open(os.path.join(UPLOAD_DIR, "README.md"), "w") as f:
            f.write(readme_content)
        print("✅ README.md created")
        
        # Step 6: Create requirements.txt
        print("\n📝 Step 6: Creating requirements.txt...")
        requirements = [
            "torch>=2.0.0",
            "transformers>=4.30.0",
            "numpy>=1.20.0",
            "scikit-learn>=1.0.0",
        ]
        with open(os.path.join(UPLOAD_DIR, "requirements.txt"), "w") as f:
            f.write("\n".join(requirements))
        print("✅ requirements.txt created")
        
        # Step 7: Create repository
        print("\n📦 Step 7: Creating HuggingFace repository...")
        api = HfApi()
        
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="model",
                exist_ok=True,
                private=False  # Set to True for private repo
            )
            print(f"✅ Repository created: {repo_id}")
        except Exception as e:
            print(f"ℹ️  Repository already exists: {e}")
        
        # Step 8: Upload files
        print("\n⬆️  Step 8: Uploading files to HuggingFace Hub...")
        print("   This may take a few minutes...")
        
        if args.create_pr:
            print("   Creating pull request...")
            pr_url = api.upload_folder(
                folder_path=UPLOAD_DIR,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload Time-RCD model",
                create_pr=True
            )
            print(f"✅ Pull request created: {pr_url}")
        else:
            api.upload_folder(
                folder_path=UPLOAD_DIR,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload Time-RCD model"
            )
            print("✅ Files uploaded successfully!")
        
        
        print("\n" + "=" * 60)
        if args.create_pr:
            print("🎉 SUCCESS! Pull request created!")
            print("=" * 60)
            print(f"\n📍 Pull Request URL:")
            print(f"   {pr_url}")
            print("\n📚 Next steps:")
            print("   1. Review the pull request")
            print("   2. Ask a maintainer to merge it")
            print("   3. Once merged, the model will be available")
        else:
            print("🎉 SUCCESS! Your model is now on HuggingFace!")
            print("=" * 60)
            print(f"\n📍 Model URL:")
            print(f"   https://huggingface.co/{repo_id}")
            print(f"\n💻 Load your model with:")
            print(f"   from transformers import AutoModel")
            print(f"   model = AutoModel.from_pretrained('{repo_id}', trust_remote_code=True)")
            print("\n📚 Next steps:")
            print("   1. Visit your model page and enhance the README")
            print("   2. Add example notebooks")
            print("   3. Tag with relevant topics")
            print("   4. Share with the community!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: Upload failed!")
        print(f"   {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
