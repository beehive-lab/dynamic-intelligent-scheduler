# TornadoVM Feature Extraction and Intelligent Execution

This tool extracts runtime and hardware features from a TornadoVM application and uses them to predict the optimal device for executing a given workload. It then runs the workload on the predicted device and reports performance. The system supports both **performance optimization** and **power optimization** modes using different ML models.

## 🚀 Overview

This project performs the following steps:

1. **Code Feature Extraction**: Runs the specified TornadoVM example to collect code-level metrics.
2. **Device Detection**: Lists all available computing devices (CPU, GPU, iGPU).
3. **Feature Parsing**: Loads extracted features from a JSON format.
4. **Device Prediction**: Predicts the best device for execution based on feature analysis.
5. **Workload Execution**: Runs the computation on the predicted optimal device.

### Optimization Modes

- **Performance Mode**: Optimizes for maximum computational speed using 3-classifier architecture
- **Power Mode**: Optimizes for energy efficiency using 6-classifier architecture (includes Java as device option)
- **Energy Mode**: Legacy energy optimization mode

---

## 🛠️ Prerequisites

Before running the script, make sure you have:

- **TornadoVM** installed with the **OpenCL backend**.
- A compatible **Java 21+ GraalVM** environment.
- Python 3 with necessary dependencies (for the extractor script).
- Your environment variables properly configured.

---

## 🔧 Environment Setup

Before running the script, configure your environment:

```bash
export JAVA_HOME=/path/to/graalvm
export PATH=/path/to/tornadovm/bin:$PATH
export TORNADO_SDK=/path/to/tornadovm/sdk
source setvars.sh
```
Replace /path/to/... with your actual local paths.

## How to Run?

To run the feature extraction and execution pipeline on a sample matrix multiplication workload:

### Performance Mode (Default)
```bash
python tornado_inference_runner -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D -s 1024 --mode performance
```

### Power Mode
```bash
python tornado_inference_runner -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D -s 1024 --mode power
```

### Energy Mode
```bash
python tornado_inference_runner -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D -s 1024 --mode energy
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `-e, --example` | TornadoVM example class to run | Required |
| `-s, --size` | Input size for the computation | Required |
| `--model-dir` | Directory containing trained models | `./ML` |
| `-f, --features-dir` | Directory where features.json will be saved | `$TORNADO_SDK` |
| `-t, --tornado-path` | Path to tornado command | `tornado` |
| `-v, --verbose` | Enable verbose output | False |
| `--mode` | Optimization mode: `performance`, `power`, or `energy` | `performance` |

## Output Summary

After running the tool, you'll see:

- Feature extraction summary
- Available devices, including GPU and iGPU details
- Predicted optimal device based on extracted features
- Actual execution on the predicted device
- Final analysis summary

## Status
This version currently operates with TornadoVM-supported backends (OpenCL tested).
It supports automatic device selection based on extracted features.

## Additional Material

- See [Developer Guide](DEVELOPER_GUIDE.md) for detailed CLI options, error handling, and integration info.
- See [Inference Engine Guide](INFERENCE_GUIDE.md) for detailed information on the prediction system.

## Acknowledgments

This work is supported by UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee for grant number 10039107 [TANGO](https://tango-project.eu).

## License
MIT
