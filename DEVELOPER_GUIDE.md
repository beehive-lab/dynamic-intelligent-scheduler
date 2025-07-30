# Detailed CLI options, error handling, and integration info of TornadoVM Feature Extractor

This script automates the process of running TornadoVM with feature extraction, predicting the optimal device using ML models, and comparing the prediction with available devices.

## Features

- **Automated Feature Extraction**: Runs TornadoVM with feature extraction enabled
- **Device Prediction**: Uses trained ML models to predict optimal hardware (CPU/GPU/iGPU)
- **Device Comparison**: Compares predicted device with actually available devices
- **Comprehensive Reporting**: Provides detailed analysis and recommendations

## Requirements

- Python 3.7+
- TornadoVM installed and accessible via `tornado` command
- Trained ML models (`.joblib` files)
- `inference_engine.py` module

## Usage

### Basic Usage

```bash
python tornado_inference_runner -e <example_class> -s <input_size>
```

### Examples

```bash
# Run MatrixMultiplication1D with size 512
python tornado_inference_runner \
  -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D \
  -s 512

# Run with custom model directory
python tornado_inference_runner \
  -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D \
  -s 1024 \
  --model-dir ./models

# Run with custom features directory
python tornado_inference_runner \
  -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D \
  -s 256 \
  -f /custom/features/path
```

### Command Line Arguments

| Argument | Description | Default           |
|----------|-------------|-------------------|
| `-e, --example` | TornadoVM example class to run | Required          |
| `-s, --size` | Input size for the computation | Required          |
| `--model-dir` | Directory containing trained models | Current directory |
| `-f, --features-dir` | Directory where features.json will be saved | `$TORNADO_SDK`    |
| `-t, --tornado-path` | Path to tornado command | `tornado`         |
| `-v, --verbose` | Enable verbose output | False             |

## How It Works

The script performs the following steps:

1. **Feature Extraction**: Runs TornadoVM with feature extraction enabled
   ```bash
   tornado --jvm="-Dtornado.feature.extraction=True -Dtornado.features.dump.dir=/path/to/features" \
           -m <example_class> <input_size>
   ```

2. **Device Discovery**: Gets available devices using `tornado --devices`

3. **Feature Loading**: Loads extracted features from the generated JSON file

4. **Device Prediction**: Uses ML models to predict optimal device (CPU/GPU/iGPU)

5. **Comparison**: Compares predicted device with available devices

6. **Reporting**: Provides detailed analysis and recommendations

## Output Example

```
======================================================================
TornadoVM FEATURE EXTRACTION AND DYNAMIC INTELLIGENT EXECUTION
======================================================================

👉 Step 1: Extract code features of selected class with TornadoVM...
✅ TornadoVM execution completed successfully
Output: Computing MxM of 1024x1024
	Single Threaded CPU Execution: 0.90 GFlops, Total time = 2397 ms
	TornadoVM Execution on GPU (Accelerated): 357.91 GFlops, Total Time = 6 ms
	Speedup: 399.5x
	Verification true



👉 Step 2: Detecting available devices...
=== Detected 1 device(s) of GPU type ===
----------------------------------------------------------------------
 Tornado device=0:0  (DEFAULT)
OPENCL --  [NVIDIA CUDA] -- NVIDIA RTX A2000 8GB Laptop GPU
Global Memory Size: 7.8 GB
Local Memory Size: 48.0 KB
Workgroup Dimensions: 3
Total Number of Block Threads: [1024]
Max WorkGroup Configuration: [1024, 1024, 64]
Device OpenCL C version: OpenCL C 1.2
----------------------------------------------------------------------
=== Detected 1 device(s) of iGPU type ===
----------------------------------------------------------------------
 Tornado device=0:1
OPENCL --  [Intel(R) OpenCL Graphics] -- Intel(R) Iris(R) Xe Graphics
Global Memory Size: 28.7 GB
Local Memory Size: 64.0 KB
Workgroup Dimensions: 3
Total Number of Block Threads: [512]
Max WorkGroup Configuration: [512, 512, 512]
Device OpenCL C version: OpenCL C 1.2
----------------------------------------------------------------------

👉 Step 3: Loading features from JSON...
✅ Features loaded successfully
   Input size: 1024
   Number of features: 29


👉 Step 4: Predicting optimal device type...
✅ Predicted optimal device type: GPU

👉 Step 5: Running workload on predicted device...
Executing: tornado --threadInfo --jvm=-Ds0.t0.device=0:0 -m tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D 1024

Computing MxM of 1024x1024
Task info: s0.t0
	Backend           : OPENCL
	Device            : NVIDIA RTX A2000 8GB Laptop GPU CL_DEVICE_TYPE_GPU (available)
	Dims              : 2
	Global work offset: [0, 0]
	Global work size  : [1024, 1024]
	Local  work size  : [32, 32, 1]
	Number of workgroups  : [32, 32]

	Single Threaded CPU Execution: 0.89 GFlops, Total time = 2403 ms
	TornadoVM Execution on GPU (Accelerated): 306.78 GFlops, Total Time = 7 ms
	Speedup: 343.2857142857143x
	Verification true


✅ Final execution completed successfully
============================================================
ANALYSIS SUMMARY
============================================================
Input size: 1024
Example class: tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D
Predicted device: GPU
Available devices: ['CPU', 'GPU', 'iGPU']
Prediction matches available: Yes

✅ Analysis completed successfully
```

## Error Handling

The script handles various error conditions:

- **TornadoVM not found**: Reports missing tornado command
- **Feature extraction failure**: Shows TornadoVM error output
- **Missing features file**: Reports file not found
- **Invalid JSON**: Reports JSON parsing errors
- **Model loading errors**: Reports missing or corrupted models
- **Device parsing errors**: Reports issues parsing device output

## Testing

Run the test script to verify everything works:

```bash
python test_tornado_extractor.py
```

This will:
- Check for required files
- Test `tornado --devices` command
- Run a complete feature extraction test
- Report success or failure

## Troubleshooting

### Common Issues

1. **TornadoVM not found**
   - Ensure TornadoVM is installed and in PATH
   - Use `-t` flag to specify custom tornado path

2. **Models not found**
   - Ensure `.joblib` files are in the model directory
   - Use `--model-dir` flag to specify model directory

3. **Features directory not writable**
   - Ensure the features directory exists and is writable
   - Use `-f` flag to specify custom features directory

4. **Permission denied**
   - Ensure the script has execute permissions: `chmod +x tornado_feature_extractor`

### Debug Mode

For detailed debugging, you can modify the script to add more verbose output or run individual components separately.

## Integration

This script can be integrated into:

- **CI/CD pipelines**: Automated testing of device predictions
- **Performance monitoring**: Regular checks of device availability
- **Development workflows**: Quick validation of ML model predictions
- **Research experiments**: Automated comparison of predictions vs reality

## Files

- `tornado_feature_extractor`: Main script
- `test_tornado_extractor.py`: Test script
- `inference_engine.py`: ML inference engine
- `*.joblib`: Trained ML models