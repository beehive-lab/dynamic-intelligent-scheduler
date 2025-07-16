# TornadoVM Feature Extraction and Intelligent Execution

This tool extracts runtime and hardware features from a TornadoVM application and uses them to predict the optimal device for executing a given workload. It then runs the workload on the predicted device and reports performance.

## 🚀 Overview

This project performs the following steps:

1. **Code Feature Extraction**: Runs the specified TornadoVM example to collect code-level metrics.
2. **Device Detection**: Lists all available computing devices (CPU, GPU, iGPU).
3. **Feature Parsing**: Loads extracted features from a JSON format.
4. **Device Prediction**: Predicts the best device for execution based on feature analysis.
5. **Workload Execution**: Runs the computation on the predicted optimal device.

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

```bash
python tornado_feature_extractor.py -e tornado.examples/uk.ac.manchester.tornado.examples.compute.MatrixMultiplication1D -s 1024
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
- See [Inference Engine Guide](INFERENCE_GUIDE) for detailed information on the prediction system.

## License
MIT