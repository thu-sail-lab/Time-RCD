"""
Test script for AnomalyCLIP HuggingFace integration

This script demonstrates proper usage of the AnomalyCLIP model with
the HuggingFace API and validates that all components work together.
"""

import numpy as np
import torch
import sys
import os

# Add the huggingface_anomalyclip module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'huggingface_anomalyclip'))

def test_processor():
    """Test the AnomalyCLIP processor"""
    print("🧪 Testing AnomalyCLIP Processor...")
    
    try:
        from processing_anomalyclip import AnomalyCLIPProcessor
        
        # Create test data
        time_series = np.random.randn(10000, 2)  # 10k samples, 2 features
        
        # Initialize processor
        processor = AnomalyCLIPProcessor(
            win_size=1000,
            stride=500,  # 50% overlap
            normalize=True,
            return_attention_mask=True
        )
        
        # Process data
        inputs = processor(time_series, return_tensors="pt")
        
        print(f"  ✅ Input shape: {time_series.shape}")
        print(f"  ✅ Output time_series shape: {inputs['time_series'].shape}")
        print(f"  ✅ Attention mask shape: {inputs['attention_mask'].shape}")
        print(f"  ✅ Processor test passed!\n")
        
        return inputs
        
    except Exception as e:
        print(f"  ❌ Processor test failed: {e}\n")
        return None

def test_config():
    """Test the AnomalyCLIP configuration"""
    print("🧪 Testing AnomalyCLIP Configuration...")
    
    try:
        from configuration_anomalyclip import AnomalyCLIPConfig
        
        # Test default config
        config = AnomalyCLIPConfig()
        print(f"  ✅ Default config created")
        print(f"  ✅ d_model: {config.d_model}")
        print(f"  ✅ num_features: {config.num_features}")
        
        # Test custom config
        custom_config = AnomalyCLIPConfig(
            d_model=1024,
            num_features=3,
            win_size=2000
        )
        print(f"  ✅ Custom config: d_model={custom_config.d_model}, num_features={custom_config.num_features}")
        print(f"  ✅ Configuration test passed!\n")
        
        return config
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}\n")
        return None

def test_model(config, inputs):
    """Test the AnomalyCLIP model"""
    print("🧪 Testing AnomalyCLIP Model...")
    
    try:
        from modeling_anomalyclip import AnomalyCLIPModel
        
        # Create model
        model = AnomalyCLIPModel(config)
        model.eval()
        
        print(f"  ✅ Model created successfully")
        print(f"  ✅ Model config: {model.config}")
        
        # Test forward pass if we have inputs
        if inputs is not None:
            with torch.no_grad():
                outputs = model(
                    time_series=inputs['time_series'],
                    attention_mask=inputs['attention_mask'],
                    return_dict=True
                )
            
            print(f"  ✅ Forward pass successful")
            print(f"  ✅ Anomaly scores shape: {outputs.anomaly_scores.shape}")
            print(f"  ✅ Reconstruction shape: {outputs.reconstruction.shape}")
            print(f"  ✅ Model test passed!\n")
            
            return model
        else:
            print(f"  ⚠️  Skipping forward pass (no inputs)")
            return model
            
    except Exception as e:
        print(f"  ❌ Model test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return None

def test_integration():
    """Test full integration"""
    print("🧪 Testing Full Integration...")
    
    try:
        # Import all components
        from configuration_anomalyclip import AnomalyCLIPConfig
        from modeling_anomalyclip import AnomalyCLIPModel
        from processing_anomalyclip import AnomalyCLIPProcessor
        
        # Create test data
        time_series = np.random.randn(5000, 1)  # Smaller for testing
        
        # Full pipeline
        processor = AnomalyCLIPProcessor(win_size=1000, normalize=True)
        config = AnomalyCLIPConfig(num_features=1)
        model = AnomalyCLIPModel(config)
        model.eval()
        
        # Preprocess
        inputs = processor(time_series, return_tensors="pt")
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract results
        scores = outputs.anomaly_scores.numpy()
        
        print(f"  ✅ Full pipeline successful!")
        print(f"  ✅ Input data: {time_series.shape}")
        print(f"  ✅ Processed: {inputs['time_series'].shape}")
        print(f"  ✅ Anomaly scores: {scores.shape}")
        print(f"  ✅ Sample scores: {scores.flat[:5]}")
        print(f"  ✅ Integration test passed!\n")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def test_model_wrapper_compatibility():
    """Test compatibility with model_wrapper.py style usage"""
    print("🧪 Testing Model Wrapper Compatibility...")
    
    def mock_run_AnomalyCLIP_HF(data, **kwargs):
        """Mock version of the model_wrapper function"""
        try:
            from configuration_anomalyclip import AnomalyCLIPConfig
            from modeling_anomalyclip import AnomalyCLIPModel
            from processing_anomalyclip import AnomalyCLIPProcessor
            
            # Extract parameters
            win_size = kwargs.get('win_size', 5000)
            batch_size = kwargs.get('batch_size', 64)
            
            # Initialize components
            processor = AnomalyCLIPProcessor(win_size=win_size, normalize=True)
            config = AnomalyCLIPConfig(num_features=data.shape[1])
            model = AnomalyCLIPModel(config)
            model.eval()
            
            # Preprocess
            inputs = processor(data, return_tensors="pt", padding=True)
            
            # Batch processing simulation
            n_windows = len(inputs['time_series'])
            scores = []
            logits = []
            
            for i in range(0, n_windows, batch_size):
                batch = {
                    k: v[i:i+batch_size] for k, v in inputs.items()
                }
                
                with torch.no_grad():
                    outputs = model(**batch)
                
                scores.append(outputs.anomaly_scores.numpy())
                logits.append(outputs.anomaly_logits.numpy())
            
            # Concatenate results
            score = np.concatenate([s.reshape(-1) for s in scores], axis=0)
            logit = np.concatenate([l.reshape(-1, 2) for l in logits], axis=0)
            
            return score, logit[:, 1] - logit[:, 0]
            
        except Exception as e:
            print(f"Mock function failed: {e}")
            raise
    
    try:
        # Test with the mock function
        test_data = np.random.randn(3000, 2)
        scores, logits = mock_run_AnomalyCLIP_HF(test_data, win_size=1000, batch_size=32)
        
        print(f"  ✅ Mock wrapper function successful!")
        print(f"  ✅ Scores shape: {scores.shape}")
        print(f"  ✅ Logits shape: {logits.shape}")
        print(f"  ✅ Sample scores: {scores[:5]}")
        print(f"  ✅ Model wrapper compatibility test passed!\n")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Model wrapper compatibility test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 AnomalyCLIP HuggingFace Integration Tests\n")
    print("=" * 60)
    
    # Test individual components
    config = test_config()
    inputs = test_processor()
    model = test_model(config, inputs)
    
    # Test integration
    integration_success = test_integration()
    wrapper_success = test_model_wrapper_compatibility()
    
    # Summary
    print("=" * 60)
    print("📊 Test Summary:")
    print(f"  Configuration: {'✅ PASS' if config else '❌ FAIL'}")
    print(f"  Processor:     {'✅ PASS' if inputs else '❌ FAIL'}")
    print(f"  Model:         {'✅ PASS' if model else '❌ FAIL'}")
    print(f"  Integration:   {'✅ PASS' if integration_success else '❌ FAIL'}")
    print(f"  Wrapper Compat: {'✅ PASS' if wrapper_success else '❌ FAIL'}")
    
    all_passed = all([config, inputs, model, integration_success, wrapper_success])
    print(f"\n🎯 Overall Result: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Your AnomalyCLIP HuggingFace integration is ready to use!")
        print("📖 See USAGE_GUIDE.md for detailed examples.")
    else:
        print("\n🔧 Please fix the failing tests before using the integration.")

if __name__ == "__main__":
    main()