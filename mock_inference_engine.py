#!/usr/bin/env python3
"""
Mock TornadoVM ML Inference Engine for demonstration
This shows the JSON parsing functionality without requiring trained models
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class MockTornadoVMInferenceEngine:
    """
    Mock inference engine for TornadoVM ML task scheduler.
    Demonstrates JSON parsing functionality without requiring trained models.
    """
    
    def __init__(self, model_dir: str = "./"):
        """
        Initialize the mock inference engine.
        """
        self.model_dir = model_dir
        
        # Define thresholds for each classifier
        self.thresholds = {
            "igpu_cpu": 0.15,    # Classifier 1 threshold
            "gpu_cpu": 0.4,      # Classifier 2 threshold  
            "gpu_igpu": 0.67     # Classifier 3 threshold
        }
        
        # Required features (same for all classifiers)
        self.required_features = [
            "threads",
            "global_memory_loads", 
            "global_memory_stores",
            "local_memory_loads",
            "local_memory_stores", 
            "total_loops",
            "parallel_loops",
            "cast_operations",
            "vector_operations",
            "total_integer_operations"
        ]
        
        # Mapping from JSON field names to required features
        self.field_mapping = {
            "Global Memory Loads": "global_memory_loads",
            "Global Memory Stores": "global_memory_stores", 
            "Local Memory Loads": "local_memory_loads",
            "Local Memory Stores": "local_memory_stores",
            "Total Loops": "total_loops",
            "Parallel Loops": "parallel_loops", 
            "Cast Operations": "cast_operations",
            "Vector Operations": "vector_operations",
            "Integer & Float Operations": "total_integer_operations"
        }
    
    def parse_json_input(self, json_data: Dict) -> Dict[str, float]:
        """
        Parse JSON input format and convert to required feature format.
        
        Args:
            json_data: Dictionary containing workload data in JSON format
            
        Returns:
            Dictionary with required features for prediction
        """
        # Extract the first workload (assuming single workload per JSON)
        workload_name = list(json_data.keys())[0]
        workload_data = json_data[workload_name]
        
        # Map JSON fields to required features
        features = {}
        
        # Handle special case for threads (not in JSON, use default or extract from workload)
        # For now, we'll use a default value or extract from workload name
        features["threads"] = 64  # Default value, could be extracted from workload name
        
        # Map the available fields
        for json_field, required_field in self.field_mapping.items():
            if json_field in workload_data:
                # Convert string to float/int
                value = workload_data[json_field]
                if isinstance(value, str):
                    features[required_field] = float(value)
                else:
                    features[required_field] = float(value)
            else:
                # Use default value if field is missing
                features[required_field] = 0.0
        
        return features
    
    def mock_predict(self, features: Dict[str, float]) -> Dict[str, any]:
        """
        Mock prediction based on feature values.
        This simulates what the real model would predict.
        """
        # Simple heuristic based on feature values
        global_memory = features.get("global_memory_loads", 0) + features.get("global_memory_stores", 0)
        local_memory = features.get("local_memory_loads", 0) + features.get("local_memory_stores", 0)
        total_ops = features.get("total_integer_operations", 0)
        threads = features.get("threads", 64)
        
        # Mock probabilities based on workload characteristics
        if global_memory > 1000 and threads > 256:
            prob_1 = 0.8  # iGPU vs CPU
            prob_2 = 0.9  # GPU vs CPU
            prob_3 = 0.7  # GPU vs iGPU
        elif local_memory > 500 and total_ops > 10000:
            prob_1 = 0.6  # iGPU vs CPU
            prob_2 = 0.8  # GPU vs CPU
            prob_3 = 0.6  # GPU vs iGPU
        else:
            prob_1 = 0.3  # iGPU vs CPU
            prob_2 = 0.4  # GPU vs CPU
            prob_3 = 0.5  # GPU vs iGPU
        
        # Apply thresholds to get binary decisions
        igpu_fit = prob_1 >= self.thresholds["igpu_cpu"]
        gpu_fit = prob_2 >= self.thresholds["gpu_cpu"]
        gpu_igpu_fit = prob_3 >= self.thresholds["gpu_igpu"]
        
        # Combine decisions to determine final device
        device_code = f"{int(igpu_fit)}{int(gpu_fit)}{int(gpu_igpu_fit)}"
        
        # Map device codes to hardware
        device_mapping = {
            '000': 'cpu', '001': 'cpu',
            '100': 'igpu', '101': 'igpu', '110': 'igpu', 
            '010': 'gpu', '011': 'gpu', '111': 'gpu'
        }
        
        predicted_device = device_mapping.get(device_code, 'cpu')
        
        return {
            "predicted_device": predicted_device,
            "confidence_scores": {
                "igpu_vs_cpu": prob_1,
                "gpu_vs_cpu": prob_2, 
                "gpu_vs_igpu": prob_3
            },
            "classifier_decisions": {
                "igpu_fit": igpu_fit,
                "gpu_fit": gpu_fit,
                "gpu_igpu_fit": gpu_igpu_fit
            },
            "raw_probabilities": [prob_1, prob_2, prob_3],
            "device_code": device_code
        }
    
    def predict_from_json(self, json_data: Dict) -> Dict[str, any]:
        """
        Predict hardware directly from JSON input format.
        
        Args:
            json_data: Dictionary containing workload data in JSON format
            
        Returns:
            Dictionary containing prediction results
        """
        # Parse JSON to required format
        features = self.parse_json_input(json_data)
        
        # Get prediction
        result = self.mock_predict(features)
        
        # Add original JSON data to result for reference
        result["input_json"] = json_data
        result["parsed_features"] = features
        
        return result
    
    def batch_predict_from_json(self, json_list: List[Dict]) -> List[Dict[str, any]]:
        """
        Predict hardware for multiple JSON inputs.
        
        Args:
            json_list: List of JSON dictionaries
            
        Returns:
            List of prediction results
        """
        results = []
        for json_data in json_list:
            try:
                result = self.predict_from_json(json_data)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        return results


# Example usage
if __name__ == "__main__":
    # Your JSON input (from features.json)
    json_input = {
        "nBody": {
            "BACKEND": "PTX",
            "DEVICE_ID": "0:2",
            "DEVICE": "GeForce GTX 1650",
            "Global Memory Loads": "15",
            "Global Memory Stores": "6",
            "Constant Memory Loads": "0",
            "Constant Memory Stores": "0",
            "Local Memory Loads": "0",
            "Local Memory Stores": "0",
            "Private Memory Loads": "20",
            "Private Memory Stores": "20",
            "Total Loops": "2",
            "Parallel Loops": "1",
            "If Statements": "2",
            "Integer Comparison": "2",
            "Float Comparison": "0",
            "Switch Statements": "0",
            "Switch Cases": "0",
            "Vector Operations": "0",
            "Integer & Float Operations": "57",
            "Boolean Operations": "9",
            "Cast Operations": "2",
            "Float Math Functions": "1",
            "Integer Math Functions": "0"
        }
    }
    
    # Initialize mock engine
    engine = MockTornadoVMInferenceEngine()
    
    # Get prediction
    result = engine.predict_from_json(json_input)
    
    # Display results
    print("Mock TornadoVM ML Prediction Results")
    print("=" * 50)
    print(f"Workload: nBody")
    print(f"Predicted Hardware: {result['predicted_device'].upper()}")
    print(f"Device Code: {result['device_code']}")
    
    print("\nConfidence Scores:")
    for classifier, score in result['confidence_scores'].items():
        print(f"  {classifier}: {score:.3f}")
    
    print("\nClassifier Decisions:")
    for decision, value in result['classifier_decisions'].items():
        status = "✓" if value else "✗"
        print(f"  {decision.replace('_', ' ').title()}: {status}")
    
    print("\nParsed Features:")
    for feature, value in result['parsed_features'].items():
        print(f"  {feature}: {value}")
    
    print("\n" + "="*50)
    print("JSON Field Mapping:")
    for json_field, required_field in engine.field_mapping.items():
        print(f"  '{json_field}' → '{required_field}'") 