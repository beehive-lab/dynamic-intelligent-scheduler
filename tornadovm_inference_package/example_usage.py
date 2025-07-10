#!/usr/bin/env python3
"""
Example usage of TornadoVM ML Inference Engine
"""

import json
from inference_engine import TornadoVMInferenceEngine

def main():
    # Initialize engine
    engine = TornadoVMInferenceEngine()
    
    # Example workloads with different characteristics
    workloads = {
        "memory_intensive": {
            "Global Memory Loads": "10000",
            "Global Memory Stores": "5000",
            "Local Memory Loads": "2000",
            "Local Memory Stores": "1000",
            "Total Loops": "1000",
            "Parallel Loops": "800",
            "Cast Operations": "50",
            "Vector Operations": "100",
            "Integer & Float Operations": "50000"
        },
        "cpu_intensive": {
            "Global Memory Loads": "100",
            "Global Memory Stores": "50",
            "Local Memory Loads": "50",
            "Local Memory Stores": "25",
            "Total Loops": "20",
            "Parallel Loops": "5",
            "Cast Operations": "5",
            "Vector Operations": "2",
            "Integer & Float Operations": "1000"
        }
    }
    
    input_sizes = [64, 256, 1024]
    
    print("TornadoVM ML Inference Engine - Examples")
    print("=" * 50)
    
    for workload_name, features in workloads.items():
        print(f"\n{workload_name.upper()}:")
        print("-" * 30)
        
        for size in input_sizes:
            # Create JSON input
            json_input = {
                workload_name: {
                    "BACKEND": "PTX",
                    "DEVICE_ID": "0:2",
                    "DEVICE": "GeForce GTX 1650",
                    **features
                }
            }
            
            # Update with input size
            json_input[workload_name]["threads"] = str(size)
            
            # Get prediction
            result = engine.predict_from_json(json_input)
            
            print(f"  Input Size {size:4d} → {result['predicted_device'].upper():4s} (Confidence: {max(result['confidence_scores'].values()):.3f})")

if __name__ == "__main__":
    main()
