#!/usr/bin/env python3
"""
TornadoVM Feature Extractor and Device Comparison Script

This script:
1. Runs TornadoVM with feature extraction enabled
2. Extracts features from the generated JSON file
3. Predicts the optimal device using the ML models
4. Compares the prediction with available devices
5. Reports any discrepancies
"""

import subprocess
import json
import os
import sys
import argparse
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional

# Import the inference engine
try:
    from inference_engine import TornadoVMInferenceEngine
except ImportError as e:
    print(f"Error importing inference_engine: {e}")
    print("Please ensure inference_engine.py is in the same directory.")
    sys.exit(1)


class TornadoFeatureExtractor:
    def __init__(self, model_dir: str = ".", tornado_path: str = "tornado"):
        """
        Initialize the feature extractor.
        
        Args:
            model_dir: Directory containing the trained models
            tornado_path: Path to the tornado command
        """
        self.model_dir = model_dir
        self.tornado_path = tornado_path
        self.inference_engine = TornadoVMInferenceEngine(model_dir)
        
    def run_tornado_with_features(self, example_class: str, input_size: int, 
                                 features_dir: str = "/home/thanos/repositories/TANGO/TornadoVM-Inference/TornadoVM/") -> bool:
        """
        Run TornadoVM with feature extraction enabled.
        
        Args:
            example_class: The TornadoVM example class to run
            input_size: Input size for the computation
            features_dir: Directory where features.json will be saved
            
        Returns:
            True if successful, False otherwise
        """
        cmd = [
            self.tornado_path,
            "--jvm=\"-Dtornado.feature.extraction=True -Dtornado.features.dump.dir=/home/thanos/repositories/TANGO/TornadoVM-Inference/tango-ml-scheduler/features.json\"",
            "-m", example_class,
            str(input_size)
        ]
        
        print(f"Running TornadoVM command:")
        print(f"  {' '.join(cmd)}")
        print()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ TornadoVM execution completed successfully")
                print(f"Output: {result.stdout}")
                if result.stderr:
                    print(f"Warnings/Errors: {result.stderr}")
                return True
            else:
                print(f"❌ TornadoVM execution failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ TornadoVM execution timed out")
            return False
        except FileNotFoundError:
            print(f"❌ TornadoVM command not found: {self.tornado_path}")
            return False
        except Exception as e:
            print(f"❌ Error running TornadoVM: {e}")
            return False
    
    def get_available_devices(self) -> Dict[str, List[str]]:
        """
        Get available TornadoVM devices.
        
        Returns:
            Dictionary mapping device types to device names
        """
        cmd = [self.tornado_path, "--devices"]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"❌ Failed to get devices: {result.stderr}")
                return {}
            
            devices = self._parse_devices_output(result.stdout)
            return devices
            
        except Exception as e:
            print(f"❌ Error getting devices: {e}")
            return {}
    
    def _parse_devices_output(self, output: str) -> Dict[str, List[str]]:
        """
        Parse the output of 'tornado --devices' command.
        
        Args:
            output: Raw output from tornado --devices
            
        Returns:
            Dictionary mapping device types to device names
        """
        devices = {
            'CPU': [],
            'GPU': [],
            'iGPU': []
        }
        
        lines = output.split('\n')
        current_device = None
        
        for line in lines:
            line = line.strip()
            
            # Look for device type indicators
            if 'OPENCL' in line and 'NVIDIA' in line:
                if 'GeForce' in line or 'RTX' in line or 'GTX' in line:
                    devices['GPU'].append(line)
                elif 'Intel' in line:
                    devices['iGPU'].append(line)
            elif 'CPU' in line and 'OpenCL' in line:
                devices['CPU'].append(line)
            elif 'CPU' in line and 'PTX' in line:
                devices['CPU'].append(line)
            elif 'Intel' in line and 'Core' in line:
                devices['CPU'].append(line)
            elif 'Intel' in line and 'Graphics' in line:
                devices['iGPU'].append(line)
            elif 'AMD' in line and 'Graphics' in line:
                devices['iGPU'].append(line)
        
        return devices
    
    def load_features_from_json(self, features_path: str) -> Optional[Dict]:
        """
        Load features from the generated JSON file.
        
        Args:
            features_path: Path to the features.json file
            
        Returns:
            Dictionary of features or None if failed
        """
        try:
            with open(features_path, 'r') as f:
                features = json.load(f)
            return features
        except FileNotFoundError:
            print(f"❌ Features file not found: {features_path}")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in features file: {e}")
            return None
        except Exception as e:
            print(f"❌ Error loading features: {e}")
            return None
    
    def predict_device(self, features: Dict) -> str:
        """
        Predict the optimal device using the ML models.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Predicted device ('CPU', 'GPU', or 'iGPU')
        """
        try:
            prediction = self.inference_engine.predict_hardware(features)
            return prediction
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return "CPU"  # Default fallback
    
    def compare_prediction_with_devices(self, predicted_device: str, 
                                      available_devices: Dict[str, List[str]]) -> Tuple[bool, str]:
        """
        Compare predicted device with available devices.
        
        Args:
            predicted_device: The predicted optimal device
            available_devices: Dictionary of available devices
            
        Returns:
            Tuple of (is_available, message)
        """
        if not available_devices:
            return False, "No devices available"

        # Map model output to available_devices keys
        device_map = {
            "cpu": "CPU",
            "gpu": "GPU",
            "igpu": "iGPU"
        }
        mapped_device = device_map.get(predicted_device.lower(), predicted_device)

        # Check if mapped device type is available
        if mapped_device in available_devices and available_devices[mapped_device]:
            return True, f"✅ Predicted device '{mapped_device}' is available"

        # Check what devices are actually available
        available_types = [device_type for device_type, devices in available_devices.items() if devices]
        
        if not available_types:
            return False, "❌ No devices are available"
        
        if len(available_types) == 1:
            return False, f"❌ Predicted '{predicted_device}' but only '{available_types[0]}' is available"
        else:
            return False, f"❌ Predicted '{predicted_device}' but available devices are: {', '.join(available_types)}"

    def normalize_features(self, features: dict) -> dict:
        mapping = {
            "Global Memory Loads": "global_memory_loads",
            "Global Memory Stores": "global_memory_stores",
            "Local Memory Loads": "local_memory_loads",
            "Local Memory Stores": "local_memory_stores",
            "Constant Memory Loads": "constant_memory_loads",
            "Constant Memory Stores": "constant_memory_stores",
            "Private Memory Loads": "private_memory_loads",
            "Private Memory Stores": "private_memory_stores",
            "Total Loops": "total_loops",
            "Parallel Loops": "parallel_loops",
            "If Statements": "if_statements",
            "Switch Statements": "switch_statements",
            "Switch Cases": "switch_cases",
            "Cast Operations": "cast_operations",
            "Vector Operations": "vector_operations",
            "Total Integer Operations": "total_integer_operations",
            "Total Float Operations": "total_float_operations",
            "Single Precision Float Operations": "single_precision_float_operations",
            "Double Precision Float Operations": "double_precision_float_operations",
            "Binary Operations": "binary_operations",
            "Boolean Operations": "boolean_operations",
            "Float Math Functions": "float_math_functions",
            "Integer Math Functions": "integer_math_functions",
            "Integer Comparison": "integer_comparison",
            "Float Comparison": "float_comparison",
            "BACKEND": "backend",
            "Device ID": "device_id",
            "threads": "threads",
            # Add more mappings as needed
        }
        BACKEND_MAP = {
            "OPENCL": 0,
            "PTX": 1,
            "SPIRV": 2,
        }
        normalized = {}
        for k, v in features.items():
            # Defensive: skip None keys
            if k is None:
                continue
            key = mapping.get(k, k)
            if key is None:
                continue
            key = str(key).lower().replace(" ", "_")
            # Special handling for backend
            if key == "backend":
                normalized[key] = BACKEND_MAP.get(str(v).upper(), -1)
            # Device ID: robustly extract a numeric value
            elif key == "device_id":
                if v is None:
                    normalized[key] = -1
                elif isinstance(v, str):
                    # Try to extract a number from the string (e.g., "0:2" -> 2)
                    import re
                    numbers = re.findall(r"\d+", v)
                    if numbers:
                        try:
                            normalized[key] = int(numbers[-1])
                        except Exception:
                            normalized[key] = -1
                    else:
                        try:
                            normalized[key] = int(v)
                        except Exception:
                            normalized[key] = -1
                else:
                    try:
                        normalized[key] = int(v)
                    except Exception:
                        normalized[key] = -1
            # Numeric features: convert to int if possible
            elif key in [
                "global_memory_loads", "global_memory_stores", "local_memory_loads", "local_memory_stores",
                "constant_memory_loads", "constant_memory_stores", "private_memory_loads", "private_memory_stores",
                "total_loops", "parallel_loops", "if_statements", "switch_statements", "switch_cases",
                "cast_operations", "vector_operations", "total_integer_operations", "total_float_operations",
                "single_precision_float_operations", "double_precision_float_operations", "binary_operations",
                "boolean_operations", "float_math_functions", "integer_math_functions", "integer_comparison",
                "float_comparison", "threads"
            ]:
                try:
                    normalized[key] = int(v)
                except Exception:
                    try:
                        normalized[key] = float(v)
                    except Exception:
                        normalized[key] = 0
            # Device: keep as string or encode if your model expects it
            elif key == "device":
                normalized[key] = str(v) if v is not None else ""
            else:
                normalized[key] = v
        return normalized

    def run_complete_analysis(self, example_class: str, input_size: int, 
                             features_dir: str = "/home/thanos/repositories/TANGO/TornadoVM-Inference/TornadoVM/") -> bool:
        """
        Run complete analysis: extract features, predict device, and compare.
        
        Args:
            example_class: The TornadoVM example class to run
            input_size: Input size for the computation
            features_dir: Directory where features.json will be saved
            
        Returns:
            True if analysis completed successfully
        """
        print("=" * 60)
        print("TORNADOVM FEATURE EXTRACTION AND DEVICE ANALYSIS")
        print("=" * 60)
        print()
        
        # Step 1: Run TornadoVM with feature extraction
        print("Step 1: Running TornadoVM with feature extraction...")
        if not self.run_tornado_with_features(example_class, input_size, features_dir):
            return False
        
        # Step 2: Get available devices
        print("\nStep 2: Getting available devices...")
        available_devices = self.get_available_devices()
        if not available_devices:
            print("❌ Could not retrieve available devices")
            return False
        
        print("Available devices:")
        for device_type, devices in available_devices.items():
            if devices:
                print(f"  {device_type}: {len(devices)} device(s)")
                for device in devices:
                    print(f"    - {device}")
        print()
        
        # # Step 3: Load features from JSON
        # print("Step 3: Loading features from JSON...")
        # #features_path = os.path.join(features_dir)
        # features_path = features_dir
        # features = self.load_features_from_json(features_path)
        # if not features:
        #     return False
        #
        # print(f"✅ Features loaded successfully")
        # print(f"   Input size: {features.get('threads', 'N/A')}")
        # print(f"   Number of features: {len(features)}")
        # print()

        # Step 3: Load features from JSON
        print("Step 3: Loading features from JSON...")
        # features_path = os.path.join(features_dir, "features.json")
        features_path = features_dir
        print("Path" + features_path)
        features_json = self.load_features_from_json(features_path)
        if features_json is None:
            print("✗ Failed to load features from JSON")
            return False

        # PATCH START: Inject threads field
        workload_name = list(features_json.keys())[0]
        features_json[workload_name]["threads"] = str(input_size)
        # PATCH END

        features_json = self.load_features_from_json(features_path)
        if features_json is None:
            print("✗ Failed to load features from JSON")
            return False

        workload_name = list(features_json.keys())[0]
        features = features_json[workload_name]
        features["threads"] = str(input_size)

        # Normalize the features dict, not the outer dict!
        features = self.normalize_features(features)

        print(f"✅ Features loaded successfully")
        print(f"   Input size: {features.get('threads', 'N/A')}")
        print(f"   Number of features: {len(features)}")
        print()

        # After normalization
        features = self.normalize_features(features)

        # Only keep features required by the model
        required_features = [
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
        filtered_features = {k: features[k] for k in required_features if k in features}

        # Now pass only filtered_features to the model
        result = self.predict_device(filtered_features)
        if isinstance(result, dict) and "predicted_device" in result:
            predicted_device = result["predicted_device"]
        else:
            predicted_device = result

        # Step 4: Predict optimal device
        print("Step 4: Predicting optimal device...")
        print(f"✅ Predicted optimal device: {predicted_device}")
        print()
        
        # Step 5: Compare prediction with available devices
        print("Step 5: Comparing prediction with available devices...")
        is_available, message = self.compare_prediction_with_devices(predicted_device, available_devices)
        print(message)
        print()
        
        # Step 6: Summary
        print("=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Input size: {input_size}")
        print(f"Example class: {example_class}")
        print(f"Predicted device: {predicted_device}")
        print(f"Available devices: {list(available_devices.keys())}")
        print(f"Prediction matches available: {'Yes' if is_available else 'No'}")
        
        if not is_available:
            print("\n⚠️  RECOMMENDATION:")
            print("The predicted device is not available. Consider:")
            print("1. Using a different device type")
            print("2. Checking if the ML models were trained on this system")
            print("3. Re-training the models with current device configuration")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="TornadoVM Feature Extractor and Device Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tornado_feature_extractor.py -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D -s 512
  python tornado_feature_extractor.py -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D -s 1024 -m ./models
  python tornado_feature_extractor.py -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D -s 256 -f /custom/features/path
        """
    )
    
    parser.add_argument(
        "-e", "--example", 
        required=True,
        help="TornadoVM example class to run"
    )
    
    parser.add_argument(
        "-s", "--size", 
        type=int, 
        required=True,
        help="Input size for the computation"
    )
    
    parser.add_argument(
        "-m", "--model-dir", 
        default=".",
        help="Directory containing trained models (default: current directory)"
    )
    
    parser.add_argument(
        "-f", "--features-dir", 
        default="/home/thanos/repositories/TANGO/TornadoVM-Inference/TornadoVM/features.json",
        help="Directory where features.json will be saved"
    )
    
    parser.add_argument(
        "-t", "--tornado-path", 
        default="tornado",
        help="Path to tornado command (default: 'tornado')"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create feature extractor
    extractor = TornadoFeatureExtractor(
        model_dir=args.model_dir,
        tornado_path=args.tornado_path
    )
    
    # Run complete analysis
    success = extractor.run_complete_analysis(
        example_class=args.example,
        input_size=args.size,
        features_dir=args.features_dir
    )
    
    if success:
        print("\n✅ Analysis completed successfully")
        sys.exit(0)
    else:
        print("\n❌ Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main() 
