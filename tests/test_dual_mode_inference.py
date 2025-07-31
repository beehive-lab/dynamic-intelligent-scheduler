#!/usr/bin/env python3
"""
Test script for TornadoVM ML Inference Engine with dual-mode support
(Performance and Power optimization modes)
"""

import json
from engine.inference_engine import TornadoVMInferenceEngine

def test_performance_mode():
    """Test performance mode inference."""
    print("=" * 60)
    print("PERFORMANCE MODE TESTING")
    print("=" * 60)
    
    try:
        engine = TornadoVMInferenceEngine(mode="performance")
        print("✅ Performance mode engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize performance engine: {e}")
        return
    
    # Test cases for performance mode
    test_cases = [
        {
            "name": "Compute-Intensive Workload",
            "features": {
                "threads": 1024,
                "global_memory_loads": 10,
                "global_memory_stores": 5,
                "local_memory_loads": 20,
                "local_memory_stores": 15,
                "total_loops": 100,
                "parallel_loops": 50,
                "cast_operations": 5,
                "vector_operations": 10,
                "total_integer_operations": 1000
            }
        },
        {
            "name": "Memory-Intensive Workload",
            "features": {
                "threads": 512,
                "global_memory_loads": 100,
                "global_memory_stores": 80,
                "local_memory_loads": 50,
                "local_memory_stores": 40,
                "total_loops": 200,
                "parallel_loops": 150,
                "cast_operations": 2,
                "vector_operations": 5,
                "total_integer_operations": 500
            }
        },
        {
            "name": "Simple Workload",
            "features": {
                "threads": 64,
                "global_memory_loads": 2,
                "global_memory_stores": 1,
                "local_memory_loads": 5,
                "local_memory_stores": 3,
                "total_loops": 10,
                "parallel_loops": 8,
                "cast_operations": 1,
                "vector_operations": 2,
                "total_integer_operations": 100
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        try:
            result = engine.predict_hardware(test_case['features'])
            
            print(f"Predicted Device: {result['predicted_device'].upper()}")
            print(f"Device Code: {result['device_code']}")
            print(f"Mode: {result['mode']}")
            
            print("Confidence Scores:")
            for classifier, score in result['confidence_scores'].items():
                print(f"  {classifier}: {score:.3f}")
                
        except Exception as e:
            print(f"❌ Test case {i} failed: {e}")

def test_power_mode():
    """Test power mode inference."""
    print("\n" + "=" * 60)
    print("POWER MODE TESTING")
    print("=" * 60)
    
    try:
        engine = TornadoVMInferenceEngine(mode="power")
        print("✅ Power mode engine initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize power engine: {e}")
        return
    
    # Test cases for power mode
    test_cases = [
        {
            "name": "High-Power Compute Workload",
            "features": {
                "threads": 1024,
                "global_memory_loads": 10,
                "global_memory_stores": 5,
                "local_memory_loads": 20,
                "local_memory_stores": 15,
                "total_loops": 100,
                "parallel_loops": 50,
                "cast_operations": 5,
                "vector_operations": 10,
                "total_integer_operations": 1000
            }
        },
        {
            "name": "Energy-Efficient Memory Workload",
            "features": {
                "threads": 512,
                "global_memory_loads": 100,
                "global_memory_stores": 80,
                "local_memory_loads": 50,
                "local_memory_stores": 40,
                "total_loops": 200,
                "parallel_loops": 150,
                "cast_operations": 2,
                "vector_operations": 5,
                "total_integer_operations": 500
            }
        },
        {
            "name": "Low-Power Simple Workload",
            "features": {
                "threads": 64,
                "global_memory_loads": 2,
                "global_memory_stores": 1,
                "local_memory_loads": 5,
                "local_memory_stores": 3,
                "total_loops": 10,
                "parallel_loops": 8,
                "cast_operations": 1,
                "vector_operations": 2,
                "total_integer_operations": 100
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_case['name']} ---")
        try:
            result = engine.predict_hardware(test_case['features'])
            
            print(f"Predicted Device: {result['predicted_device'].upper()}")
            print(f"Base Device: {result['base_device'].upper()}")
            print(f"Java Recommended: {result['java_recommended']}")
            print(f"Device Code: {result['device_code']}")
            print(f"Mode: {result['mode']}")
            
            print("Confidence Scores:")
            for classifier, score in result['confidence_scores'].items():
                print(f"  {classifier}: {score:.3f}")
                
            print("Classifier Decisions:")
            for decision, value in result['classifier_decisions'].items():
                status = "✓" if value else "✗"
                print(f"  {decision}: {status}")
                
        except Exception as e:
            print(f"❌ Test case {i} failed: {e}")

def test_json_input():
    """Test JSON input format for both modes."""
    print("\n" + "=" * 60)
    print("JSON INPUT TESTING")
    print("=" * 60)
    
    json_input = {
        "nBody": {
            "BACKEND": "PTX",
            "DEVICE_ID": "0:2",
            "DEVICE": "GeForce GTX 1650",
            "Global Memory Loads": "15",
            "Global Memory Stores": "6",
            "Local Memory Loads": "0",
            "Local Memory Stores": "0",
            "Total Loops": "2",
            "Parallel Loops": "1",
            "Cast Operations": "2",
            "Vector Operations": "0",
            "Integer & Float Operations": "57"
        }
    }
    
    # Test performance mode with JSON
    print("\n--- Performance Mode with JSON ---")
    try:
        engine_perf = TornadoVMInferenceEngine(mode="performance")
        result_perf = engine_perf.predict_from_json(json_input)
        print(f"✅ Performance Mode - Predicted Device: {result_perf['predicted_device'].upper()}")
        print(f"   Mode: {result_perf['mode']}")
    except Exception as e:
        print(f"❌ Performance mode JSON test failed: {e}")
    
    # Test power mode with JSON
    print("\n--- Power Mode with JSON ---")
    try:
        engine_power = TornadoVMInferenceEngine(mode="power")
        result_power = engine_power.predict_from_json(json_input)
        print(f"✅ Power Mode - Predicted Device: {result_power['predicted_device'].upper()}")
        print(f"   Base Device: {result_power['base_device'].upper()}")
        print(f"   Java Recommended: {result_power['java_recommended']}")
        print(f"   Mode: {result_power['mode']}")
    except Exception as e:
        print(f"❌ Power mode JSON test failed: {e}")

def test_batch_processing():
    """Test batch processing for both modes."""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING TESTING")
    print("=" * 60)
    
    # Create multiple test cases
    test_features = [
        {
            "threads": 1024,
            "global_memory_loads": 10,
            "global_memory_stores": 5,
            "local_memory_loads": 20,
            "local_memory_stores": 15,
            "total_loops": 100,
            "parallel_loops": 50,
            "cast_operations": 5,
            "vector_operations": 10,
            "total_integer_operations": 1000
        },
        {
            "threads": 512,
            "global_memory_loads": 100,
            "global_memory_stores": 80,
            "local_memory_loads": 50,
            "local_memory_stores": 40,
            "total_loops": 200,
            "parallel_loops": 150,
            "cast_operations": 2,
            "vector_operations": 5,
            "total_integer_operations": 500
        },
        {
            "threads": 64,
            "global_memory_loads": 2,
            "global_memory_stores": 1,
            "local_memory_loads": 5,
            "local_memory_stores": 3,
            "total_loops": 10,
            "parallel_loops": 8,
            "cast_operations": 1,
            "vector_operations": 2,
            "total_integer_operations": 100
        }
    ]
    
    # Test performance mode batch
    print("\n--- Performance Mode Batch Processing ---")
    try:
        engine_perf = TornadoVMInferenceEngine(mode="performance")
        results_perf = engine_perf.batch_predict(test_features)
        
        for i, result in enumerate(results_perf, 1):
            if "error" in result:
                print(f"❌ Case {i}: {result['error']}")
            else:
                print(f"✅ Case {i}: {result['predicted_device'].upper()} (Mode: {result['mode']})")
                
    except Exception as e:
        print(f"❌ Performance batch test failed: {e}")
    
    # Test power mode batch
    print("\n--- Power Mode Batch Processing ---")
    try:
        engine_power = TornadoVMInferenceEngine(mode="power")
        results_power = engine_power.batch_predict(test_features)
        
        for i, result in enumerate(results_power, 1):
            if "error" in result:
                print(f"❌ Case {i}: {result['error']}")
            else:
                print(f"✅ Case {i}: {result['predicted_device'].upper()} (Base: {result['base_device'].upper()}, Java: {result['java_recommended']})")
                
    except Exception as e:
        print(f"❌ Power batch test failed: {e}")

def main():
    """Run all tests."""
    print("TornadoVM ML Inference Engine - Dual Mode Testing")
    print("=" * 80)
    
    # Test performance mode
    test_performance_mode()
    
    # Test power mode
    test_power_mode()
    
    # Test JSON input
    test_json_input()
    
    # Test batch processing
    test_batch_processing()
    
    print("\n" + "=" * 80)
    print("All tests completed!")

if __name__ == "__main__":
    main() 