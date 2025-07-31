#!/usr/bin/env python3
"""
Simple example demonstrating TornadoVM ML Inference Engine with dual-mode support
(Performance and Power optimization modes)
"""

from engine.inference_engine import TornadoVMInferenceEngine

def main():
    print("TornadoVM ML Inference Engine - Dual Mode Example")
    print("=" * 60)
    
    # Example workload features
    workload_features = {
        "threads": 1024,
        "global_memory_loads": 50,
        "global_memory_stores": 25,
        "local_memory_loads": 30,
        "local_memory_stores": 15,
        "total_loops": 100,
        "parallel_loops": 80,
        "cast_operations": 10,
        "vector_operations": 20,
        "total_integer_operations": 2000
    }
    
    print(f"Workload Features:")
    for feature, value in workload_features.items():
        print(f"  {feature}: {value}")
    
    print("\n" + "=" * 60)
    
    # Test Performance Mode
    print("1. PERFORMANCE MODE ANALYSIS")
    print("-" * 40)
    try:
        engine_perf = TornadoVMInferenceEngine(mode="performance")
        result_perf = engine_perf.predict_hardware(workload_features)
        
        print(f"✅ Predicted Device: {result_perf['predicted_device'].upper()}")
        print(f"   Mode: {result_perf['mode']}")
        print(f"   Device Code: {result_perf['device_code']}")
        
        print("\n   Confidence Scores:")
        for classifier, score in result_perf['confidence_scores'].items():
            print(f"     {classifier}: {score:.3f}")
            
    except Exception as e:
        print(f"❌ Performance mode failed: {e}")
    
    print("\n" + "=" * 60)
    
    # Test Power Mode
    print("2. POWER MODE ANALYSIS")
    print("-" * 40)
    try:
        engine_power = TornadoVMInferenceEngine(mode="power")
        result_power = engine_power.predict_hardware(workload_features)
        
        print(f"✅ Predicted Device: {result_power['predicted_device'].upper()}")
        print(f"   Mode: {result_power['mode']}")
        print(f"   Base Device: {result_power['base_device'].upper()}")
        print(f"   Java Recommended: {result_power['java_recommended']}")
        print(f"   Device Code: {result_power['device_code']}")
        
        print("\n   Confidence Scores:")
        for classifier, score in result_power['confidence_scores'].items():
            print(f"     {classifier}: {score:.3f}")
            
        print("\n   Classifier Decisions:")
        for decision, value in result_power['classifier_decisions'].items():
            status = "✓" if value else "✗"
            print(f"     {decision}: {status}")
            
    except Exception as e:
        print(f"❌ Power mode failed: {e}")
    
    print("\n" + "=" * 60)
    
    # Test JSON Input
    print("3. JSON INPUT TEST")
    print("-" * 40)
    
    json_input = {
        "matrixMultiplication": {
            "BACKEND": "PTX",
            "DEVICE_ID": "0:2",
            "DEVICE": "GeForce GTX 1650",
            "Global Memory Loads": "1000",
            "Global Memory Stores": "500",
            "Local Memory Loads": "200",
            "Local Memory Stores": "100",
            "Total Loops": "100",
            "Parallel Loops": "80",
            "Cast Operations": "10",
            "Vector Operations": "20",
            "Integer & Float Operations": "2000"
        }
    }
    
    try:
        result_json_perf = engine_perf.predict_from_json(json_input)
        print(f"✅ Performance Mode (JSON): {result_json_perf['predicted_device'].upper()}")
        
        result_json_power = engine_power.predict_from_json(json_input)
        print(f"✅ Power Mode (JSON): {result_json_power['predicted_device'].upper()}")
        
    except Exception as e:
        print(f"❌ JSON test failed: {e}")
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("\nUsage:")
    print("  # Performance mode")
    print("  python tornado_inference_runner -e <example> -s <size> --mode performance")
    print("  # Power mode")
    print("  python tornado_inference_runner -e <example> -s <size> --mode power")
    print("  # Test inference engine directly")
    print("  python tests/test_dual_mode_inference.py")

if __name__ == "__main__":
    main() 