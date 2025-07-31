# Dual-Mode Architecture: Performance vs Power Optimization

This document explains the enhanced TornadoVM ML Inference Engine that supports both **performance optimization** and **power optimization** modes using different machine learning architectures.

## Overview

The inference engine now supports three optimization modes:

1. **Performance Mode** (Default): 3-classifier architecture for speed optimization
2. **Power Mode** (New): 6-classifier architecture for energy efficiency
3. **Energy Mode** (Legacy): Original energy optimization mode

## Architecture Comparison

### Performance Mode (3-Classifiers)

**Goal**: Maximize computational speed

**Classifiers**:
- `clf1`: iGPU vs CPU (classification)
- `clf2`: GPU vs CPU (classification) 
- `clf3`: GPU vs iGPU (classification)

**Decision Logic**:
```
Device Code = [clf1][clf2][clf3]
000, 001 → CPU
100, 101, 110 → iGPU  
010, 011, 111 → GPU
```

**Thresholds**:
- `igpu_cpu`: 0.15
- `gpu_cpu`: 0.4
- `gpu_igpu`: 0.67

### Power Mode (6-Classifiers)

**Goal**: Minimize energy consumption

**Classifiers**:
- `clf1`: iGPU vs CPU (regression)
- `clf2`: GPU vs CPU (classification)
- `clf3`: GPU vs iGPU (classification)
- `clf4`: Java vs CPU (classification)
- `clf5`: Java vs GPU (classification)
- `clf6`: Java vs iGPU (classification)

**Decision Logic**:
```
1. Base Device Selection:
   Device Code = [clf1][clf2][clf3]
   000, 001 → CPU base
   100, 101, 110 → iGPU base
   010, 011, 111 → GPU base

2. Java Comparison:
   If base_device == "cpu" and clf4 > threshold → Java
   If base_device == "gpu" and clf5 > threshold → Java  
   If base_device == "igpu" and clf6 > threshold → Java
   Otherwise → base_device
```

**Thresholds**:
- `igpu_cpu`: 0.0 (regression threshold)
- `gpu_cpu`: 0.4
- `gpu_igpu`: 0.67
- `java_cpu`: 0.5
- `java_gpu`: 0.5
- `java_igpu`: 0.5

## Key Differences

| Aspect | Performance Mode | Power Mode |
|--------|------------------|------------|
| **Classifiers** | 3 | 6 |
| **Device Options** | CPU, iGPU, GPU | CPU, iGPU, GPU, Java |
| **clf1 Type** | Classification | Regression |
| **Java Support** | No | Yes |
| **Optimization Target** | Speed | Energy Efficiency |
| **Model Files** | `.joblib` | `.pkl` |

## Model Files

### Performance Mode
```
ML/Performance-Trained-Models/
├── IGPUvsCPU_final.joblib
├── GPUvsCPU_final.joblib
└── GPUvsIGPU_final.joblib
```

### Power Mode
```
ML/Power-trained-Models/
├── clf1_energy_etc.pkl
├── clf2_energy_etc.pkl
├── clf3_energy_etc.pkl
├── clf4_energy_etc.pkl
├── clf5_energy_etc.pkl
└── clf6_energy_etc.pkl
```

## Usage Examples

### Command Line

```bash
# Performance mode (default)
python tornado_inference_runner -e <example> -s <size> --mode performance

# Power mode
python tornado_inference_runner -e <example> -s <size> --mode power

# Energy mode (legacy)
python tornado_inference_runner -e <example> -s <size> --mode energy
```

### Python API

```python
from engine.inference_engine import TornadoVMInferenceEngine

# Performance mode
engine_perf = TornadoVMInferenceEngine(mode="performance")
result_perf = engine_perf.predict_hardware(features)

# Power mode
engine_power = TornadoVMInferenceEngine(mode="power")
result_power = engine_power.predict_hardware(features)
```

## Output Format

### Performance Mode Output
```python
{
    "predicted_device": "gpu",
    "confidence_scores": {
        "igpu_vs_cpu": 0.85,
        "gpu_vs_cpu": 0.92,
        "gpu_vs_igpu": 0.78
    },
    "classifier_decisions": {
        "igpu_fit": True,
        "gpu_fit": True,
        "gpu_igpu_fit": True
    },
    "raw_probabilities": [0.85, 0.92, 0.78],
    "device_code": "111",
    "mode": "performance"
}
```

### Power Mode Output
```python
{
    "predicted_device": "java",
    "base_device": "cpu",
    "java_recommended": True,
    "confidence_scores": {
        "igpu_vs_cpu": 0.3,
        "gpu_vs_cpu": 0.2,
        "gpu_vs_igpu": 0.4,
        "java_vs_cpu": 0.8,
        "java_vs_gpu": 0.6,
        "java_vs_igpu": 0.7
    },
    "classifier_decisions": {
        "igpu_fit": False,
        "gpu_fit": False,
        "gpu_igpu_fit": False,
        "java_cpu_fit": True,
        "java_gpu_fit": True,
        "java_igpu_fit": True
    },
    "raw_probabilities": [0.3, 0.2, 0.4, 0.8, 0.6, 0.7],
    "device_code": "000",
    "mode": "power"
}
```

## Testing

### Run Comprehensive Tests
```bash
python tests/test_dual_mode_inference.py
```

### Run Simple Example
```bash
python example_dual_mode.py
```

### Test Individual Modes
```bash
# Performance mode test
python tests/simple_json_example.py

# Power mode test (if models available)
python engine/inference_engine.py
```

## Model Training

The power mode models were trained using:
- **Dataset**: Energy consumption data from TornadoVM workloads
- **Features**: 10 runtime features (threads, memory ops, loops, etc.)
- **Architecture**: 6-classifier ensemble with different thresholds
- **Validation**: Cross-validation with energy efficiency metrics

## Performance Characteristics

### Power Mode Advantages
- **Energy Efficiency**: Optimizes for minimal power consumption
- **Java Integration**: Considers Java as a viable device option
- **Fine-grained Control**: 6-classifier architecture provides more nuanced decisions
- **Regression Support**: clf1 uses regression for continuous energy prediction

### Performance Mode Advantages
- **Speed**: Optimized for maximum computational throughput
- **Simplicity**: 3-classifier architecture is faster to evaluate
- **Maturity**: Well-tested with extensive validation
- **Compatibility**: Works with existing TornadoVM workflows

## Migration Guide

### From Performance to Power Mode
1. Ensure power model files are available in `ML/Power-trained-Models/`
2. Change mode parameter: `mode="power"`
3. Update output parsing to handle additional fields (`base_device`, `java_recommended`)
4. Consider Java device availability in your system

### From Energy to Power Mode
1. Power mode is the successor to energy mode
2. Enhanced with Java device support
3. More sophisticated 6-classifier architecture
4. Better energy efficiency predictions

## Troubleshooting

### Common Issues

1. **Missing Power Models**: Ensure `.pkl` files are in `ML/Power-trained-Models/`
2. **Mode Mismatch**: Verify mode parameter matches available models
3. **Java Device**: Power mode may recommend Java - ensure Java runtime is available
4. **Threshold Tuning**: Power mode thresholds can be adjusted in `inference_engine.py`

### Error Messages

- `❌ Could not find model file`: Check model directory and file names
- `❌ Power mode failed`: Verify power model files exist and are compatible
- `❌ Performance mode failed`: Check performance model files

## Future Enhancements

- **Adaptive Thresholds**: Dynamic threshold adjustment based on workload characteristics
- **Multi-objective Optimization**: Simultaneous performance and power optimization
- **Real-time Learning**: Online model updates based on execution feedback
- **Device-specific Models**: Specialized models for different hardware configurations 