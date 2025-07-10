#!/usr/bin/env python3
"""
Example usage of TornadoVM ML Inference Engine
"""

from inference_engine import TornadoVMInferenceEngine

def main():
    # Initialize the inference engine
    engine = TornadoVMInferenceEngine()
    
    # Example 1: Memory-intensive workload (likely GPU)
    memory_intensive = {
        "threads": 1024,
        "global_memory_loads": 50000,
        "global_memory_stores": 25000,
        "local_memory_loads": 1000,
        "local_memory_stores": 500,
        "total_loops": 1000,
        "parallel_loops": 800,
        "cast_operations": 50,
        "vector_operations": 100,
        "total_integer_operations": 100000
    }
    
    # Example 2: CPU-intensive workload (likely CPU)
    cpu_intensive = {
        "threads": 8,
        "global_memory_loads": 100,
        "global_memory_stores": 50,
        "local_memory_loads": 200,
        "local_memory_stores": 100,
        "total_loops": 100,
        "parallel_loops": 10,
        "cast_operations": 20,
        "vector_operations": 5,
        "total_integer_operations": 50000
    }
    
    # Example 3: Balanced workload (might be iGPU)
    balanced_workload = {
        "threads": 256,
        "global_memory_loads": 5000,
        "global_memory_stores": 2500,
        "local_memory_loads": 500,
        "local_memory_stores": 250,
        "total_loops": 200,
        "parallel_loops": 150,
        "cast_operations": 30,
        "vector_operations": 25,
        "total_integer_operations": 25000
    }
    
    # Test predictions
    workloads = {
        "Memory-Intensive": memory_intensive,
        "CPU-Intensive": cpu_intensive,
        "Balanced": balanced_workload
    }
    
    print("TornadoVM ML Inference Engine - Hardware Predictions")
    print("=" * 60)
    
    for name, features in workloads.items():
        try:
            result = engine.predict_hardware(features)
            
            print(f"\n{name} Workload:")
            print(f"  Predicted Device: {result['predicted_device'].upper()}")
            print(f"  Confidence Scores:")
            print(f"    iGPU vs CPU: {result['confidence_scores']['igpu_vs_cpu']:.3f}")
            print(f"    GPU vs CPU:  {result['confidence_scores']['gpu_vs_cpu']:.3f}")
            print(f"    GPU vs iGPU: {result['confidence_scores']['gpu_vs_igpu']:.3f}")
            print(f"  Device Code: {result['device_code']}")
            
        except Exception as e:
            print(f"\n{name} Workload: ERROR - {e}")
    
    # Show feature importance
    print("\n" + "=" * 60)
    print("Feature Importance (iGPU vs CPU classifier):")
    importance = engine.get_feature_importance("igpu_cpu")
    features = engine.required_features
    
    for i, (feature, importance_score) in enumerate(zip(features, importance["igpu_cpu"])):
        print(f"  {i+1:2d}. {feature:25s}: {importance_score:.4f}")

if __name__ == "__main__":
    main() 