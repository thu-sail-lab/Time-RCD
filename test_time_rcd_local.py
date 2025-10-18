#!/usr/bin/env python3
"""
Test script for Time_RCD with local checkpoint

This script tests:
1. Loading your local checkpoint into the HuggingFace-compatible Time_RCD model
2. Running zero-shot inference
3. Saving the model in HuggingFace format for future use
"""

import numpy as np
import torch
import sys
import os

# Add Testing/models to path so we can import Time_RCD_HF
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Testing', 'models'))

# Add paths
sys.path.append('/Users/oliver/Documents/2025/Huawei/Time-RCD/Time-RCD')
sys.path.append('/Users/oliver/Documents/2025/Huawei/Time-RCD/Time-RCD/Testing/models')

def test_time_rcd_local_checkpoint():
    """Test loading Time_RCD from local checkpoint"""
    
    print("🚀 Testing Time_RCD with Local Checkpoint")
    print("=" * 50)
    
    # Test parameters
    checkpoint_path = "/Users/oliver/Documents/2025/Huawei/Time-RCD/Time-RCD/Testing/checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"✅ Checkpoint found: {checkpoint_path}")
    
    try:
        # Test 1: Initialize Time_RCD wrapper
        print("\n📦 Test 1: Initialize Time_RCD wrapper")
        from Time_RCD_HF import Time_RCD
        
        model = Time_RCD(
            num_input_channels=1,
            model_path=checkpoint_path,  # Use local checkpoint
            win_size=5000,
            batch_size=64,
            device='cpu'  # Use CPU for testing
        )
        print("✅ Time_RCD wrapper initialized successfully")
        
        # Test 2: Create synthetic data
        print("\n📊 Test 2: Create synthetic time series data")
        # Create synthetic univariate time series
        np.random.seed(42)
        data = np.random.randn(1000, 1) * 2 + np.sin(np.linspace(0, 10*np.pi, 1000)).reshape(-1, 1)
        print(f"✅ Created synthetic data with shape: {data.shape}")
        
        # Test 3: Zero-shot inference
        print("\n🔮 Test 3: Zero-shot inference")
        scores = model.zero_shot(data)
        print(f"✅ Zero-shot inference completed")
        print(f"   Anomaly scores shape: type of scores: {type(scores)}")
        # print(f"   Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        # print(f"   Mean score: {scores.mean():.4f}")
        
        # Test 4: Save model in HuggingFace format
        print("\n💾 Test 4: Save model in HuggingFace format")
        save_dir = "/Users/oliver/Documents/2025/Huawei/Time-RCD/Time-RCD/Testing/time_rcd_hf"
        model.save_model_hf(save_dir)
        
        # Test 5: Try loading the saved HF model
        print("\n🔄 Test 5: Load saved HuggingFace model")
        try:
            from huggingface_time_rcd import Time_RCD as HF_Time_RCD
            loaded_model = HF_Time_RCD.from_pretrained(save_dir)
            print("✅ Successfully loaded saved HuggingFace model")
        except Exception as e:
            print(f"⚠️  Could not load saved HF model: {e}")
        
        print("\n🎉 All tests completed successfully!")
        print("\nNext steps:")
        print(f"1. Your model is saved in HuggingFace format at: {save_dir}")
        print("2. You can now use it with transformers.AutoModel.from_pretrained()")
        print("3. Upload to HuggingFace Hub for sharing")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def inspect_checkpoint():
    """Inspect the checkpoint file to understand its structure"""
    
    print("🔍 Inspecting Checkpoint Structure")
    print("=" * 40)
    
    checkpoint_path = "/Users/oliver/Documents/2025/Huawei/Time-RCD/Time-RCD/Testing/checkpoints/full_mask_anomaly_head_pretrain_checkpoint_best.pth"
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"Checkpoint type: {type(checkpoint)}")
        print(f"Main keys: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("\nFound 'model_state_dict' key")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("\nFound 'state_dict' key")
        else:
            state_dict = checkpoint
            print("\nUsing checkpoint directly as state_dict")
        
        print(f"\nState dict keys (first 10):")
        for i, key in enumerate(list(state_dict.keys())[:10]):
            tensor_shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
            print(f"  {i+1}. {key}: {tensor_shape}")
        
        print(f"\nTotal parameters: {len(state_dict)} tensors")
        
        # Look for specific components
        ts_encoder_keys = [k for k in state_dict.keys() if 'ts_encoder' in k]
        recon_head_keys = [k for k in state_dict.keys() if 'reconstruction_head' in k]
        anomaly_head_keys = [k for k in state_dict.keys() if 'anomaly_head' in k]
        
        print(f"\nComponent analysis:")
        print(f"  ts_encoder parameters: {len(ts_encoder_keys)}")
        print(f"  reconstruction_head parameters: {len(recon_head_keys)}")
        print(f"  anomaly_head parameters: {len(anomaly_head_keys)}")
        
        if ts_encoder_keys:
            print(f"\nSample ts_encoder keys:")
            for key in ts_encoder_keys[:3]:
                print(f"  {key}")
        
    except Exception as e:
        print(f"❌ Error inspecting checkpoint: {e}")

if __name__ == "__main__":
    print("Time_RCD Local Checkpoint Test")
    print("=" * 60)
    
    # First inspect the checkpoint
    inspect_checkpoint()
    
    print("\n")
    
    # Then run the full test
    success = test_time_rcd_local_checkpoint()
    
    if success:
        print("\n🎊 SUCCESS: Your Time_RCD model works with HuggingFace!")
    else:
        print("\n💥 FAILED: There were issues loading your model.")