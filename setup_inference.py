#!/usr/bin/env python3
"""
Setup script for TornadoVM ML Inference Engine
Creates a deployable package with all required files
"""

import os
import shutil
import json
import subprocess
import sys

def create_deployment_package():
    """Create a complete deployment package for inference."""
    
    print("Creating TornadoVM ML Inference Engine Deployment Package")
    print("=" * 60)
    
    # Create deployment directory
    deployment_dir = "tornadovm_inference_package"
    if os.path.exists(deployment_dir):
        shutil.rmtree(deployment_dir)
    os.makedirs(deployment_dir)
    
    # Files to copy
    files_to_copy = [
        "inference_engine.py",
        "engine.py", 
        "utilities.py",
        "plotting_utilities.py",
        "requirements.txt",
        "README_inference.md",
        "features.json",
        "IGPUvsCPU_final.joblib",
        "GPUvsCPU_final.joblib", 
        "GPUvsIGPU_final.joblib",
        "Final Artifacts/features.txt"
    ]
    
    # Copy files
    print("Copying required files...")
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            # Create subdirectories if needed
            dest_path = os.path.join(deployment_dir, file_path)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.copy2(file_path, dest_path)
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (not found)")
    
    # Create a simple test script
    test_script = """#!/usr/bin/env python3
\"\"\"
Quick test script for TornadoVM ML Inference Engine
\"\"\"

import sys
import os

# Add current directory to path
sys.path.append('.')

try:
    from inference_engine import TornadoVMInferenceEngine
    
    # Test with sample data
    json_input = {
        "testWorkload": {
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
    
    print("Testing TornadoVM ML Inference Engine...")
    engine = TornadoVMInferenceEngine()
    result = engine.predict_from_json(json_input)
    
    print("✓ Test successful!")
    print(f"Predicted Hardware: {result['predicted_device'].upper()}")
    print(f"Confidence: {max(result['confidence_scores'].values()):.3f}")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    sys.exit(1)
"""
    
    with open(os.path.join(deployment_dir, "test_inference.py"), "w") as f:
        f.write(test_script)
    
    # Create installation script
    install_script = """#!/bin/bash
# TornadoVM ML Inference Engine Installation Script

echo "Installing TornadoVM ML Inference Engine..."
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Test the installation
echo "Testing inference engine..."
python3 test_inference.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Installation completed successfully!"
    echo ""
    echo "Usage examples:"
    echo "  python3 engine.py features.json 64"
    echo "  python3 engine.py -v features.json 1024"
    echo "  python3 test_inference.py"
else
    echo "✗ Installation test failed"
    exit 1
fi
"""
    
    with open(os.path.join(deployment_dir, "install.sh"), "w") as f:
        f.write(install_script)
    os.chmod(os.path.join(deployment_dir, "install.sh"), 0o755)
    
    # Create a comprehensive README
    readme_content = """# TornadoVM ML Inference Engine

A machine learning-based hardware prediction system for TornadoVM computational tasks.

## Quick Start

1. **Install dependencies:**
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

2. **Test the installation:**
   ```bash
   python3 test_inference.py
   ```

3. **Run inference:**
   ```bash
   python3 engine.py features.json 64
   ```

## Usage

### Command Line Interface

```bash
python3 engine.py <json_file> <input_size>
```

**Examples:**
```bash
# Basic usage
python3 engine.py features.json 64

# Verbose output
python3 engine.py -v features.json 1024

# Different input sizes
python3 engine.py features.json 256
python3 engine.py features.json 2048
```

### JSON Input Format

Your JSON file should contain workload features:

```json
{
    "workloadName": {
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
```

**Note:** The `threads` field will be automatically set to your input size.

### Output

The engine predicts the optimal hardware (CPU, iGPU, or GPU) and provides:

- **Predicted Hardware:** CPU, iGPU, or GPU
- **Confidence Scores:** From each classifier
- **Device Code:** 3-digit binary code representing decisions
- **Reasoning:** Explanation based on input size

### Input Size Guidelines

| Input Size | Typical Prediction | Reasoning |
|------------|-------------------|-----------|
| < 128      | CPU               | Small workloads favor CPU |
| 128-1024   | iGPU              | Medium workloads favor iGPU |
| > 1024     | GPU               | Large workloads favor GPU |

## Files Included

- `inference_engine.py` - Core inference engine
- `engine.py` - Command-line interface
- `utilities.py` - Utility functions
- `plotting_utilities.py` - Visualization utilities
- `requirements.txt` - Python dependencies
- `*.joblib` - Trained model files
- `Final Artifacts/features.txt` - Feature definitions
- `features.json` - Example input file
- `test_inference.py` - Quick test script
- `install.sh` - Installation script

## Requirements

- Python 3.7+
- pip3
- Internet connection (for dependency installation)

## Troubleshooting

1. **"Module not found" errors:**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **"Model files not found" errors:**
   Make sure all `.joblib` files are in the current directory.

3. **"JSON file not found" errors:**
   Check that your JSON file exists and is readable.

## Integration

To integrate with TornadoVM:

```bash
# In your TornadoVM script
python3 engine.py workload_features.json $INPUT_SIZE
```

The engine will return the predicted hardware that you can use for scheduling.

## Support

For issues or questions, check the model training notebook or contact the development team.
"""
    
    with open(os.path.join(deployment_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    # Create a simple example script
    example_script = """#!/usr/bin/env python3
\"\"\"
Example usage of TornadoVM ML Inference Engine
\"\"\"

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
        print(f"\\n{workload_name.upper()}:")
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
"""
    
    with open(os.path.join(deployment_dir, "example_usage.py"), "w") as f:
        f.write(example_script)
    
    print("✓ Deployment package created successfully!")
    print(f"Package location: {deployment_dir}/")
    
    # Create a compressed archive
    print("\\nCreating compressed archive...")
    try:
        shutil.make_archive("tornadovm_inference_package", "zip", deployment_dir)
        print("✓ Compressed archive created: tornadovm_inference_package.zip")
    except Exception as e:
        print(f"✗ Failed to create archive: {e}")
    
    # Print summary
    print("\\n" + "=" * 60)
    print("DEPLOYMENT PACKAGE SUMMARY")
    print("=" * 60)
    print(f"Package directory: {deployment_dir}/")
    print("Compressed archive: tornadovm_inference_package.zip")
    print("")
    print("Files included:")
    for file_path in files_to_copy:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
    
    print("")
    print("Additional files created:")
    print("  ✓ test_inference.py")
    print("  ✓ install.sh")
    print("  ✓ README.md")
    print("  ✓ example_usage.py")
    
    print("")
    print("To deploy on another system:")
    print("1. Copy the package directory or zip file")
    print("2. Run: chmod +x install.sh && ./install.sh")
    print("3. Test with: python3 test_inference.py")
    print("4. Use with: python3 engine.py features.json 64")

if __name__ == "__main__":
    create_deployment_package() 