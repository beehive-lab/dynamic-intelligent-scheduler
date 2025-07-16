#!/usr/bin/env python3
"""
Test script for TornadoVM Feature Extractor

This script demonstrates how to use the tornado_feature_extractor.py
to run TornadoVM with feature extraction and compare predictions with available devices.
"""

import subprocess
import sys
import os

def test_tornado_devices():
    """Test if tornado --devices works"""
    print("Testing tornado --devices command...")
    try:
        result = subprocess.run(["tornado", "--devices"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ tornado --devices works")
            print("Available devices:")
            print(result.stdout)
            return True
        else:
            print(f"❌ tornado --devices failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ tornado command not found")
        return False
    except Exception as e:
        print(f"❌ Error testing tornado --devices: {e}")
        return False

def test_feature_extractor():
    """Test the feature extractor with a simple example"""
    print("\n" + "="*60)
    print("TESTING TORNADOVM FEATURE EXTRACTOR")
    print("="*60)
    features_json_file=os.path.join(os.environ["TORNADO_SDK"], "features.json")
    feature_file_arg = (
        f"-f={features_json_file}"
    )
    # Test command
    cmd = [
        "python", "tornado_feature_extractor.py",
        "-e", "tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D",
        "-s", "512",
        "-m", ".",
        feature_file_arg
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Feature extractor test completed successfully")
            return True
        else:
            print(f"❌ Feature extractor test failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Feature extractor test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running feature extractor test: {e}")
        return False

def main():
    print("TORNADOVM FEATURE EXTRACTOR TEST")
    print("="*40)
    
    # Check if required files exist
    required_files = [
        "tornado_feature_extractor.py",
        "inference_engine.py",
        "IGPUvsCPU_final.joblib",
        "GPUvsCPU_final.joblib", 
        "GPUvsIGPU_final.joblib"
    ]
    
    print("Checking required files...")
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            return False
    
    print()
    
    # Test tornado devices
    if not test_tornado_devices():
        print("⚠️  Warning: tornado --devices failed, but continuing...")
    
    # Test feature extractor
    success = test_feature_extractor()
    
    if success:
        print("\n🎉 All tests completed successfully!")
        print("\nYou can now use the feature extractor with:")
        print("python tornado_feature_extractor.py -e <example_class> -s <size>")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 