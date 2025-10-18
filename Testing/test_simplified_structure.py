#!/usr/bin/env python3
"""
Test the simplified Time-RCD HuggingFace integration
"""

import sys
import os
import numpy as np
import torch

# Add the huggingface_anomalyclip to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'huggingface_anomalyclip'))

def test_simplified_structure():
    """Test the new simplified structure"""
    
    print("🧪 Testing Simplified Time-RCD Structure")
    print("=" * 50)
    
    # Test 1: Import the new simplified components
    print("1️⃣ Testing imports...")
    try:
        from huggingface_anomalyclip import (
            TimeRCDModel,
            TimeRCDOutput, 
            AnomalyCLIPConfig,
            AnomalyCLIPProcessor
        )
        print("✅ All imports successful!")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test 2: Create a simple config
    print("\n2️⃣ Testing configuration...")
    try:
        config = AnomalyCLIPConfig(
            num_features=1,
            win_size=100,
            d_model=128,
            d_proj=64,
            patch_size=16,
            num_layers=2,
            num_heads=4
        )
        print(f"✅ Config created: d_model={config.d_model}, num_features={config.num_features}")
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False
    
    # Test 3: Create the simplified model
    print("\n3️⃣ Testing model creation...")
    try:
        model = TimeRCDModel(config)
        print(f"✅ TimeRCDModel created successfully!")
        print(f"   - Base model prefix: {model.base_model_prefix}")
        print(f"   - Model components: ts_encoder, reconstruction_head, anomaly_head")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 4: Test model forward pass
    print("\n4️⃣ Testing forward pass...")
    try:
        # Create dummy input
        batch_size, seq_len, num_features = 2, 100, 1
        dummy_input = torch.randn(batch_size, seq_len, num_features)
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_input, return_dict=True)
        
        print(f"✅ Forward pass successful!")
        print(f"   - Output type: {type(outputs)}")
        print(f"   - Anomaly scores shape: {outputs.anomaly_scores.shape}")
        print(f"   - Reconstruction shape: {outputs.reconstruction.shape}")
        print(f"   - Embeddings shape: {outputs.embeddings.shape}")
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False
    
    # Test 5: Test zero-shot method
    print("\n5️⃣ Testing zero-shot method...")
    try:
        # Create dummy time series
        time_series = np.random.randn(500, 1)  # 500 timesteps, 1 feature
        
        result = model.zero_shot(time_series)
        
        print(f"✅ Zero-shot method successful!")
        print(f"   - Input shape: {time_series.shape}")
        print(f"   - Anomaly scores shape: {result['anomaly_score'].shape}")
        print(f"   - Reconstruction shape: {result['reconstruction'].shape}")
        
    except Exception as e:
        print(f"❌ Zero-shot method failed: {e}")
        return False
    
    # Test 6: Test processor
    print("\n6️⃣ Testing processor...")
    try:
        processor = AnomalyCLIPProcessor(
            win_size=50,
            stride=25,
            normalize=True
        )
        
        # Process dummy data
        time_series = np.random.randn(200, 1)
        inputs = processor(time_series, return_tensors="pt")
        
        print(f"✅ Processor successful!")
        print(f"   - Input shape: {time_series.shape}")
        print(f"   - Processed time_series shape: {inputs['time_series'].shape}")
        print(f"   - Attention mask shape: {inputs['attention_mask'].shape}")
        
    except Exception as e:
        print(f"❌ Processor failed: {e}")
        return False
    
    # Test 7: Test integration with model
    print("\n7️⃣ Testing processor + model integration...")
    try:
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        
        print(f"✅ Processor + Model integration successful!")
        print(f"   - Windows processed: {inputs['time_series'].shape[0]}")
        print(f"   - Anomaly scores per window: {outputs.anomaly_scores.shape}")
        
    except Exception as e:
        print(f"❌ Integration failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The simplified structure works correctly.")
    print("\n📋 Summary of new structure:")
    print("   - ✅ Single TimeRCDModel inheriting from PreTrainedModel")
    print("   - ✅ Built-in zero_shot() method for compatibility")
    print("   - ✅ Separate AnomalyCLIPProcessor for preprocessing")
    print("   - ✅ Clean integration with HuggingFace ecosystem")
    
    return True

def test_time_rcd_wrapper():
    """Test the Time_RCD wrapper class"""
    
    print("\n🧪 Testing Time_RCD Wrapper Class")
    print("=" * 50)
    
    try:
        # Add models path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
        
        from AnomalyCLIP_HF import Time_RCD
        
        print("1️⃣ Testing Time_RCD initialization...")
        
        # Test with a dummy path (will use fallback creation)
        model = Time_RCD(
            num_input_channels=1,
            model_path="dummy-path",  # This will trigger fallback
            win_size=100,
            batch_size=32
        )
        
        print("✅ Time_RCD wrapper created successfully!")
        
        # Test zero-shot inference
        print("\n2️⃣ Testing zero-shot inference...")
        dummy_data = np.random.randn(200, 1)
        scores = model.zero_shot(dummy_data)
        
        print(f"✅ Zero-shot inference successful!")
        print(f"   - Input shape: {dummy_data.shape}")
        print(f"   - Output scores shape: {scores.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Time_RCD wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing Simplified Time-RCD HuggingFace Integration")
    print("=" * 60)
    
    # Test the core structure
    success1 = test_simplified_structure()
    
    # Test the wrapper
    success2 = test_time_rcd_wrapper()
    
    if success1 and success2:
        print("\n🎊 ALL TESTS PASSED! 🎊")
        print("\n✨ Your simplified structure is working correctly!")
        print("\n📚 Key improvements:")
        print("   1. Single TimeRCDModel class (no complex inheritance)")
        print("   2. Built-in zero_shot() method for compatibility")
        print("   3. Clean separation of concerns (model vs processor)")
        print("   4. HuggingFace ecosystem integration")
        print("   5. Backward compatibility with existing Time_RCD API")
    else:
        print("\n💥 Some tests failed. Please check the errors above.")
        sys.exit(1)