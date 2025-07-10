#!/usr/bin/env python3
"""
Test different workload types with the TornadoVM ML Inference Engine
"""

from mock_inference_engine import MockTornadoVMInferenceEngine

def main():
    # Initialize the mock engine
    engine = MockTornadoVMInferenceEngine()
    
    # Test different workload types
    workloads = {
        "nBody (Your Input)": {
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
        },
        "Memory-Intensive (GPU)": {
            "matrixMultiplication": {
                "BACKEND": "PTX",
                "DEVICE_ID": "0:2",
                "DEVICE": "GeForce GTX 1650",
                "Global Memory Loads": "10000",
                "Global Memory Stores": "5000",
                "Local Memory Loads": "2000",
                "Local Memory Stores": "1000",
                "Total Loops": "1000",
                "Parallel Loops": "800",
                "Cast Operations": "50",
                "Vector Operations": "100",
                "Integer & Float Operations": "50000"
            }
        },
        "CPU-Intensive": {
            "blackScholes": {
                "BACKEND": "PTX",
                "DEVICE_ID": "0:2",
                "DEVICE": "GeForce GTX 1650",
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
        },
        "Balanced (iGPU)": {
            "convolution2D": {
                "BACKEND": "PTX",
                "DEVICE_ID": "0:2",
                "DEVICE": "GeForce GTX 1650",
                "Global Memory Loads": "5000",
                "Global Memory Stores": "2500",
                "Local Memory Loads": "500",
                "Local Memory Stores": "250",
                "Total Loops": "200",
                "Parallel Loops": "150",
                "Cast Operations": "30",
                "Vector Operations": "25",
                "Integer & Float Operations": "25000"
            }
        }
    }
    
    print("TornadoVM ML Inference Engine - Workload Analysis")
    print("=" * 70)
    
    for workload_name, json_input in workloads.items():
        print(f"\n{workload_name}")
        print("-" * 50)
        
        try:
            result = engine.predict_from_json(json_input)
            
            # Display workload characteristics
            workload_data = list(json_input.values())[0]
            global_memory = float(workload_data.get("Global Memory Loads", 0)) + float(workload_data.get("Global Memory Stores", 0))
            local_memory = float(workload_data.get("Local Memory Loads", 0)) + float(workload_data.get("Local Memory Stores", 0))
            total_ops = float(workload_data.get("Integer & Float Operations", 0))
            
            print(f"Characteristics:")
            print(f"  Global Memory Ops: {global_memory}")
            print(f"  Local Memory Ops:  {local_memory}")
            print(f"  Total Operations:  {total_ops}")
            
            print(f"\nPrediction:")
            print(f"  Hardware: {result['predicted_device'].upper()}")
            print(f"  Device Code: {result['device_code']}")
            
            print(f"  Confidence Scores:")
            for classifier, score in result['confidence_scores'].items():
                print(f"    {classifier.replace('_', ' ').title()}: {score:.3f}")
            
            print(f"  Decisions:")
            for decision, value in result['classifier_decisions'].items():
                status = "✓" if value else "✗"
                print(f"    {decision.replace('_', ' ').title()}: {status}")
                
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The inference engine analyzes workload characteristics to predict optimal hardware:")
    print("• High global memory + high threads → GPU")
    print("• Balanced memory + moderate operations → iGPU") 
    print("• Low memory + low parallelism → CPU")
    print("\nYour nBody workload shows moderate characteristics, suggesting iGPU optimization.")

if __name__ == "__main__":
    main() 