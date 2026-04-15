# Weed Enumerations

This directory contains serialization-friendly enumerations used throughout the Weed library for type identification, device management, and module classification.

## Overview

The `enums/` directory defines a comprehensive set of enumerations that serve as type tags for serialization, device selection, and module identification. These enums are designed to be stable across library versions and suitable for binary serialization.

## Enumerations

### [`device_tag.hpp`](device_tag.hpp) - Device Types

Defines the available hardware device backends:



**Usage:**
- Selects the computation backend for tensors and storage
- Used in tensor construction: `Tensor(data, shape, requires_grad, device, device_id)`
- Determines whether operations use CPU or GPU acceleration

### [`dtype.hpp`](dtype.hpp) - Data Types

Defines the supported data types for tensors:



**Usage:**
- Specifies the element type of tensors
- `REAL`: Used for neural network weights, activations, inputs
- `COMPLEX`: Used for quantum computing simulations
- `INT`: Used exclusively for `SymbolTensor` (token indices, embeddings)

### [`storage_type.hpp`](storage_type.hpp) - Storage Backends

Defines all possible storage configurations:



**Usage:**
- Identifies storage backend for serialization
- Determines memory layout (dense vs sparse)
- Specifies device location (CPU vs GPU)
- Used in `Storage::load()` to reconstruct storage type

### [`module_type.hpp`](module_type.hpp) - Neural Network Module Types

Defines 33 module types for serialization and type identification:

```cpp
enum ModuleType {
    NONE_MODULE_TYPE = 0,
    SEQUENTIAL_T = 1,          // Sequential container
    LINEAR_T = 2,              // Fully connected layer
    RELU_T = 3,                // ReLU activation
    SIGMOID_T = 4,             // Sigmoid activation
    TANH_T = 5,                // Tanh activation
    DROPOUT_T = 6,             // Dropout regularization
    LAYERNORM_T = 7,           // Layer normalization
    EMBEDDING_T = 8,           // Embedding lookup
    GRU_T = 9,                 // Gated Recurrent Unit
    LSTM_T = 10,               // Long Short-Term Memory
    MIGRATE_CPU_T = 11,        // CPU migration module
    MIGRATE_GPU_T = 12,        // GPU migration module
    SOFTMAX_T = 13,            // Softmax activation
    LOGSOFTMAX_T = 14,         // Log-softmax activation
    QRACK_NEURON_T = 15,       // Quantum neuron
    QRACK_NEURON_LAYER_T = 16, // Quantum neuron layer
    MULTIHEAD_ATTENTION_T = 17,// Multi-head attention
    TRANSFORMER_ENCODER_LAYER_T = 18, // Transformer encoder
    GELU_T = 19,               // Gaussian Error Linear Unit
    MEAN_T = 20,               // Mean reduction
    MIN_T = 21,                // Min reduction
    MAX_T = 22,                // Max reduction
    RESHAPE_T = 23,            // Reshape operation
    VARIANCE_T = 24,           // Variance reduction
    STDDEV_T = 25,             // Standard deviation
    POSITIONAL_ENCODING_T = 26,// Sinusoidal positional encoding
    MEAN_CENTER_T = 27,        // Mean centering
    FLATTEN_T = 28,            // Flatten operation
    LEARNED_POSITIONAL_ENCODING_T = 29, // Learnable positional encoding
    RMS_NORM_T = 30,           // RMS normalization
    ROPE_T = 31,               // Rotary positional embedding
    SWIGLU_T = 32,             // SwiGLU activation
    QWEN_DECODER_LAYER_T = 33  // Qwen decoder layer
    QWEN_T = 34                // Qwen model wrapper
};
```


**Module Categories:**

| Category | Types |
|----------|-------|
| **Containers** | `SEQUENTIAL_T` |
| **Basic Layers** | `LINEAR_T`, `EMBEDDING_T` |
| **Activations** | `RELU_T`, `SIGMOID_T`, `TANH_T`, `GELU_T`, `SWIGLU_T` |
| **RNN** | `GRU_T`, `LSTM_T` |
| **Attention** | `MULTIHEAD_ATTENTION_T`, `TRANSFORMER_ENCODER_LAYER_T` |
| **Normalization** | `LAYERNORM_T`, `RMS_NORM_T`, `DROPOUT_T` |
| **Positional** | `POSITIONAL_ENCODING_T`, `LEARNED_POSITIONAL_ENCODING_T`, `ROPE_T` |
| **Quantum** | `QRACK_NEURON_T`, `QRACK_NEURON_LAYER_T` |
| **Utilities** | `MEAN_T`, `MAX_T`, `MIN_T`, `VARIANCE_T`, `STDDEV_T`, `RESHAPE_T`, `FLATTEN_T` |
| **Migration** | `MIGRATE_CPU_T`, `MIGRATE_GPU_T` |
| **LLM-specific** | `QWEN_DECODER_LAYER_T` |

**Usage:**
- Serialization: `Module::write_module_type()` / `read_module_type()`
- Type identification during deserialization
- Module factory pattern for reconstruction

### [`activation_function_type.hpp`](activation_function_type.hpp) - Activation Functions

Defines activation function types:



**Usage:**
- Configures activation in modules (e.g., `Linear` with activation)
- Serialization of activation parameters
- Activation function selection in model construction

### [`quantum_function_type.hpp`](quantum_function_type.hpp) - Quantum Circuit Functions

Defines quantum circuit function types for Qrack integration:



**Usage:**
- Configures quantum operations in quantum modules
- Serialization of quantum circuit specifications
- Qrack integration for quantum-classical hybrid models

## Serialization Design

### Stability Guarantees

1. **Fixed Values**: Enum values are explicitly assigned and stable
2. **No Reordering**: New types are appended, not inserted
3. **Reserved Values**: `NONE_*` values reserved for "none" state
4. **Type Safety**: Strong typing prevents invalid values

### Serialization Format



### Deserialization Safety

- Unknown enum values should be handled gracefully
- `NONE_*` values indicate absence of type
- Type checking before casting to concrete module types

## Usage Patterns

### Device Selection



### Module Serialization



### Storage Creation



## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).
