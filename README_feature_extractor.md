# TornadoVM Feature Extractor

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
python tornado_feature_extractor.py -e <example_class> -s <input_size>
```

### Examples

```bash
# Run MatrixMultiplication1D with size 512
python tornado_feature_extractor.py \
  -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D \
  -s 512

# Run with custom model directory
python tornado_feature_extractor.py \
  -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D \
  -s 1024 \
  -m ./models

# Run with custom features directory
python tornado_feature_extractor.py \
  -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D \
  -s 256 \
  -f /custom/features/path
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-e, --example` | TornadoVM example class to run | Required |
| `-s, --size` | Input size for the computation | Required |
| `-m, --model-dir` | Directory containing trained models | Current directory |
| `-f, --features-dir` | Directory where features.json will be saved | `/home/mikepapadim/manchester/TornadoVM/` |
| `-t, --tornado-path` | Path to tornado command | `tornado` |
| `-v, --verbose` | Enable verbose output | False |

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
============================================================
TORNADOVM FEATURE EXTRACTION AND DEVICE ANALYSIS
============================================================

Step 1: Running TornadoVM with feature extraction...
Running TornadoVM command:
  tornado --jvm=-Dtornado.feature.extraction=True -Dtornado.features.dump.dir=/home/mikepapadim/manchester/TornadoVM/ -m tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D 512

✅ TornadoVM execution completed successfully

Step 2: Getting available devices...
Available devices:
  GPU: 1 device(s)
    - OPENCL --  [NVIDIA CUDA] -- NVIDIA GeForce RTX 3070

Step 3: Loading features from JSON...
✅ Features loaded successfully
   Input size: 512
   Number of features: 15

Step 4: Predicting optimal device...
✅ Predicted optimal device: GPU

Step 5: Comparing prediction with available devices...
✅ Predicted device 'GPU' is available

============================================================
ANALYSIS SUMMARY
============================================================
Input size: 512
Example class: tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D
Predicted device: GPU
Available devices: ['GPU']
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
   - Use `-m` flag to specify model directory

3. **Features directory not writable**
   - Ensure the features directory exists and is writable
   - Use `-f` flag to specify custom features directory

4. **Permission denied**
   - Ensure the script has execute permissions: `chmod +x tornado_feature_extractor.py`

### Debug Mode

For detailed debugging, you can modify the script to add more verbose output or run individual components separately.

## Integration

This script can be integrated into:

- **CI/CD pipelines**: Automated testing of device predictions
- **Performance monitoring**: Regular checks of device availability
- **Development workflows**: Quick validation of ML model predictions
- **Research experiments**: Automated comparison of predictions vs reality

## Files

- `tornado_feature_extractor.py`: Main script
- `test_tornado_extractor.py`: Test script
- `inference_engine.py`: ML inference engine
- `*.joblib`: Trained ML models
- `README_feature_extractor.md`: This documentation 