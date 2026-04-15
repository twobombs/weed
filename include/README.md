# Weed Public API Headers

This directory contains the public header files for the Weed library, organized by functional module. Weed is a minimalist AI/ML inference and backpropagation library designed in the style of Qrack, supporting both classical and quantum computing paradigms.

## Library Overview

Weed provides:
- **Tensor operations** with automatic differentiation (autograd)
- **Neural network modules** for building deep learning models
- **Multi-device support** (CPU via OpenCL, with quantum simulator support via Qrack)
- **Flexible precision** (half, float, double, or float128 via `WEED_FPPOW`)
- **Sparse tensor support** for memory-efficient computations

## Configuration

The library is configured via CMake with these key options:
- `WEED_FPPOW`: Floating-point precision (4=half, 5=float, 6=double, 7=float128)
- `WEED_TCAPPOW`: Tensor capacity integer size (3-7 for 8-128 bit)
- `WEED_ENABLE_OPENCL`: Enable GPU acceleration
- `QRACK_AVAILABLE`: Enable quantum computing support

## Directory Structure

### [`common/`](common/) - Core Utilities and Type Definitions

The `common/` directory contains fundamental type definitions and utility functions used throughout the library:

- **[`weed_types.hpp`](common/weed_types.hpp)**: Defines core type aliases including:
  - `real1`: Configurable floating-point type (half, float, double, or float128)
  - `complex`: Complex number type (`std::complex<real1>`)
  - `tcapint`, `symint`, `tlenint`: Configurable integer types for tensor indexing
  - `RealPtr`, `ComplexPtr`, `IntPtr`: Smart pointers for aligned memory allocation
  - Mathematical constants (`PI_R1`, `E_R1`, `REAL1_EPSILON`)
  - Sparse vector types (`RealSparseVector`, `ComplexSparseVector`)

- **[`weed_functions.hpp`](common/weed_functions.hpp)**: Utility functions including:
  - `log2Gpu()`: Fast log2 computation using CPU intrinsics when available
  - `pow2Gpu()`: Power of 2 computation
  - `cl_alloc()`, `cl_free()`: OpenCL memory allocation wrappers

- **[`oclapi.hpp`](common/oclapi.hpp)**: OpenCL API wrapper definitions for kernel dispatch

- **[`oclengine.hpp`](common/oclengine.hpp)**: OpenCL engine singleton for device management

- **[`parallel_for.hpp`](common/parallel_for.hpp)**: Parallel execution utilities

- **[`serializer.hpp`](common/serializer.hpp)**: Serialization framework for saving/loading models

- **[`half.hpp`](common/half.hpp)**: Half-precision floating-point implementation

### [`devices/`](devices/) - Hardware Device Abstraction

The `devices/` directory provides abstraction for GPU/OpenCL device management:

- **[`gpu_device.hpp`](devices/gpu_device.hpp)**: Core GPU device management class (`GpuDevice`):
  - Manages OpenCL context, command queue, and device-specific resources
  - Handles memory allocation tracking and VRAM limits
  - Implements kernel dispatch queue with callback cycle
  - Provides buffer operations (read/write, map/unmap, fill, clear)
  - Manages dependent events for ordered kernel execution
  - Key methods: `MakeBuffer()`, `QueueCall()`, `RequestKernel()`, `LockSync()`, `UnlockSync()`

- **[`pool_item.hpp`](devices/pool_item.hpp)**: OpenCL kernel argument buffer pool:
  - `PoolItem`: Pre-allocated buffers for complex and VCI arguments
  - Reduces allocation overhead during kernel dispatch
  - Includes custom `bad_alloc` exception with descriptive messages

- **[`queue_item.hpp`](devices/queue_item.hpp)**: Kernel call request wrapper:
  - `QueueItem`: Structures kernel call parameters before pool assignment
  - Stores API call type, work item counts, local group sizes, and buffers

### [`enums/`](enums/) - Type Enumerations

The `enums/` directory defines serialization-friendly enumerations:

- **[`device_tag.hpp`](enums/device_tag.hpp)**: Device types
  - `NONE_DEVICE`, `DEFAULT_DEVICE`, `CPU`, `GPU`

- **[`dtype.hpp`](enums/dtype.hpp)**: Data types
  - `NONE_DTYPE`, `REAL`, `COMPLEX`, `INT`, `DEFAULT_DTYPE`

- **[`storage_type.hpp`](enums/storage_type.hpp)**: Storage backends
  - `NONE_STORAGE_TYPE`, `REAL_CPU_DENSE`, `REAL_GPU_DENSE`, `COMPLEX_CPU_DENSE`, `COMPLEX_GPU_DENSE`, `INT_CPU_DENSE`, `INT_GPU_DENSE`, `REAL_CPU_SPARSE`, `COMPLEX_CPU_SPARSE`

- **[`module_type.hpp`](enums/module_type.hpp)**: Neural network module types (33 types):
  - Basic: `SEQUENTIAL_T`, `LINEAR_T`, `RELU_T`, `SIGMOID_T`, `TANH_T`
  - Advanced: `GRU_T`, `LSTM_T`, `MULTIHEAD_ATTENTION_T`, `TRANSFORMER_ENCODER_LAYER_T`
  - Normalization: `DROPOUT_T`, `LAYERNORM_T`, `RMS_NORM_T`
  - Embedding: `EMBEDDING_T`, `POSITIONAL_ENCODING_T`, `LEARNED_POSITIONAL_ENCODING_T`
  - Quantum: `QRACK_NEURON_T`, `QRACK_NEURON_LAYER_T`
  - Utilities: `MEAN_T`, `MAX_T`, `MIN_T`, `SOFTMAX_T`, `LOGSOFTMAX_T`, `GELU_T`
  - LLM-specific: `ROPE_T`, `SWIGLU_T`, `QWEN_DECODER_LAYER_T`, `QWEN_MODEL_T`

- **[`activation_function_type.hpp`](enums/activation_function_type.hpp)**: Activation functions
  - `NONE_FN`, `SIGMOID_FN`, `TANH_FN`, `RELU_FN`, `GELU_FN`, `SWIGLU_FN`

- **[`quantum_function_type.hpp`](enums/quantum_function_type.hpp)**: Quantum circuit functions
  - `CUSTOM_QFN`, `NONE_QFN`, `BELL_GHZ_QFN`, `ALT_BELL_GHZ_QFN`, `QFT_QFN`, `IQFT_QFN`

### [`storage/`](storage/) - Memory Management Abstraction

The `storage/` directory implements the storage abstraction layer for tensor data:

- **[`storage.hpp`](storage/storage.hpp)**: Base `Storage` class:
  - Abstract interface for all storage backends
  - Tracks storage type, device, data type, and element count
  - Pure virtual methods: `FillZeros()`, `FillOnes()`, `Upcast()`, `is_gpu()`, `cpu()`, `gpu()`
  - Serialization support via `save()` and static `load()`

- **[`typed_storage.hpp`](storage/typed_storage.hpp)**: Template-based typed storage:
  - `TypedStorage<T>`: Template specialization for `real1`, `complex`, `tcapint`
  - Type-specific operations: `operator[]`, `write()`, `add()`, `FillValue()`
  - Aligned memory allocation with platform-specific optimizations
  - Type aliases: `IntStorage`, `RealStorage`, `ComplexStorage`

- **[`all_storage.hpp`](storage/all_storage.hpp)**: Factory function for storage creation

- **CPU Storage Implementations**:
  - **[`cpu_storage.hpp`](storage/cpu_storage.hpp)**: Base CPU storage
  - **[`cpu_real_storage.hpp`](storage/cpu_real_storage.hpp)**: Real-valued CPU storage
  - **[`cpu_complex_storage.hpp`](storage/cpu_complex_storage.hpp)**: Complex-valued CPU storage
  - **[`cpu_int_storage.hpp`](storage/cpu_int_storage.hpp)**: Integer CPU storage

- **GPU Storage Implementations**:
  - **[`gpu_storage.hpp`](storage/gpu_storage.hpp)**: Base GPU storage
  - **[`gpu_real_storage.hpp`](storage/gpu_real_storage.hpp)**: Real-valued GPU storage
  - **[`gpu_complex_storage.hpp`](storage/gpu_complex_storage.hpp)**: Complex-valued GPU storage
  - **[`gpu_int_storage.hpp`](storage/gpu_int_storage.hpp)**: Integer GPU storage

- **Sparse Storage Implementations**:
  - **[`sparse_cpu_storage.hpp`](storage/sparse_cpu_storage.hpp)**: Base sparse CPU storage
  - **[`sparse_cpu_real_storage.hpp`](storage/sparse_cpu_real_storage.hpp)**: Real-valued sparse storage
  - **[`sparse_cpu_complex_storage.hpp`](storage/sparse_cpu_complex_storage.hpp)**: Complex-valued sparse storage

### [`tensors/`](tensors/) - Tensor Abstraction

The `tensors/` directory defines the core tensor interface:

- **[`base_tensor.hpp`](tensors/base_tensor.hpp)**: Base tensor class:
  - Shape and stride management
  - Offset tracking for views
  - Reshape, transpose, flatten operations
  - Contiguity checking

- **[`tensor.hpp`](tensors/tensor.hpp)**: Main `Tensor` class with autograd:
  - Arbitrary dimensions with shape/stride arrays
  - `requires_grad` flag for gradient tracking
  - `grad_node` and `grad` for backpropagation
  - Device placement (CPU/GPU)
  - Static factory methods: `zeros()`, `ones_like()`, `one_hot()`, `make_gradient()`
  - Tensor operations: `add()`, `mul()`, `matmul()`, `sub()`, `div()`, `pow()`, `exp()`, `log()`
  - Activation functions: `sigmoid()`, `tanh()`, `relu()`, `gelu()`, `softmax()`, `logsoftmax()`
  - Reduction operations: `sum()`, `mean()`, `variance()`, `stddev()`, `max()`, `min()`, `abs()`
  - Utility operations: `clamp()`, `sin()`, `cos()`, `slice()`, `chunk()`, `contiguous()`, `reshape()`, `transpose()`, `flatten()`
  - Operator overloading for natural syntax: `+`, `-`, `*`, `/`, `>>`, `<<`, `^`

- **[`real_tensor.hpp`](tensors/real_tensor.hpp)**: `RealTensor` specialization:
  - Direct access to real-valued elements via `operator[]`, `write()`, `add()`
  - Type-safe interface for real tensors

- **[`symbol_tensor.hpp`](tensors/symbol_tensor.hpp)**: `SymbolTensor` for indexing:
  - Non-mathematical tensor for integer enumeration (e.g., token indices)
  - Supports reshape, transpose, flatten operations
  - Used for embedding layer inputs

- **[`parameter.hpp`](tensors/parameter.hpp)**: Trainable parameter wrapper:
  - Wraps tensors with trainable flags
  - Integration with optimizer systems

### [`modules/`](modules/) - Neural Network Building Blocks

The `modules/` directory contains composable neural network layers:

- **[`module.hpp`](modules/module.hpp)**: Base `Module` class:
  - `forward()` method for inference
  - `parameters()` method for trainable weights
  - `train()`/`eval()` mode switching
  - Serialization support via `save()` and static `load()`
  - Quantum interface support via `Qrack::QInterfacePtr`

- **Core Layers**:
  - **[`linear.hpp`](modules/linear.hpp)** / **[`linear.cpp`](../src/modules/linear.cpp)**: Fully connected layer
  - **[`sequential.hpp`](modules/sequential.hpp)** / **[`sequential.cpp`](../src/modules/sequential.cpp)**: Container for sequential models
  - **[`dropout.hpp`](modules/dropout.hpp)** / **[`dropout.cpp`](../src/modules/dropout.cpp)**: Dropout regularization
  - **[`layernorm.hpp`](modules/layernorm.hpp)** / **[`layernorm.cpp`](../src/modules/layernorm.cpp)**: Layer normalization
  - **[`embedding.hpp`](modules/embedding.hpp)** / **[`embedding.cpp`](../src/ops/embedding.cpp)**: Embedding lookup layer

- **RNN Variants**:
  - **[`gru.hpp`](modules/gru.hpp)** / **[`gru.cpp`](../src/modules/gru.cpp)**: Gated Recurrent Unit
  - **[`lstm.hpp`](modules/lstm.hpp)** / **[`lstm.cpp`](../src/modules/lstm.cpp)**: Long Short-Term Memory

- **Attention Mechanisms**:
  - **[`multihead_attention.hpp`](modules/multihead_attention.hpp)** / **[`multihead_attention.cpp`](../src/modules/multihead_attention.cpp)**: Multi-head self-attention
  - **[`positional_encoding.hpp`](modules/positional_encoding.hpp)** / **[`positional_encoding.cpp`](../src/modules/positional_encoding.cpp)**: Sinusoidal positional encoding
  - **[`learned_positional_encoding.hpp`](modules/learned_positional_encoding.hpp)** / **[`learned_positional_encoding.cpp`](../src/modules/learned_positional_encoding.cpp)**: Learnable positional embeddings
  - **[`rope.hpp`](modules/rope.hpp)** / **[`rope.cpp`](../src/modules/rope.cpp)**: Rotary Positional Embedding (RoPE)

- **Transformer Components**:
  - **[`transformer_encoder_layer.hpp`](modules/transformer_encoder_layer.hpp)** / **[`transformer_encoder_layer.cpp`](../src/modules/transformer_encoder_layer.cpp)**: Transformer encoder layer
  - **[`qwen_decoder_layer.hpp`](modules/qwen_decoder_layer.hpp)** / **[`qwen_decoder_layer.cpp`](../src/modules/qwen_decoder_layer.cpp)**: Qwen-style decoder layer
  - **[`qwen_model.hpp`](modules/qwen_model.hpp)** / **[`qwen_model.cpp`](../src/modules/qwen_model.cpp)**: Qwen complete model architecture
  - **[`qwen_tokenizer.hpp`](modules/qwen_tokenizer.hpp)** / **[`qwen_tokenizer.cpp`](../src/modules/qwen_tokenizer.cpp)**: Qwen byte-level BPE tokenizer
  - **[`swiglu.hpp`](modules/swiglu.hpp)** / **[`swiglu.cpp`](../src/modules/swiglu.cpp)**: SwiGLU activation

- **Normalization**:
  - **[`rms_norm.hpp`](modules/rms_norm.hpp)**: RMS normalization

- **Quantum Modules**:
  - **[`qrack_neuron.hpp`](modules/qrack_neuron.hpp)** / **[`qrack_neuron.cpp`](../src/modules/qrack_neuron.cpp)**: Quantum neuron layer
  - **[`qrack_neuron_layer.hpp`](modules/qrack_neuron_layer.hpp)** / **[`qrack_neuron_layer.cpp`](../src/modules/qrack_neuron_layer.cpp)**: Quantum neuron layer container

- **Utilities**:
  - **[`migrate_cpu.hpp`](modules/migrate_cpu.hpp)** / **[`migrate_cpu.cpp`](../src/modules/migrate_cpu.cpp)**: CPU migration module
  - **[`migrate_gpu.hpp`](modules/migrate_gpu.hpp)** / **[`migrate_gpu.cpp`](../src/modules/migrate_gpu.cpp)**: GPU migration module

### [`ops/`](ops/) - Tensor Operations

The `ops/` directory implements individual tensor operations with autograd support:

- **Element-wise Operations**:
  - **[`abs.hpp`](ops/abs.hpp)** / **[`abs.cpp`](../src/ops/abs.cpp)**: Absolute value
    - **[`clamp.hpp`](ops/clamp.hpp)** / **[`clamp.cpp`](../src/ops/clamp.cpp)**: Value clamping
    - **[`div.hpp`](ops/div.hpp)** / **[`div.cpp`](../src/ops/div.cpp)**: Element-wise division
    - **[`pow.hpp`](ops/pow.hpp)** / **[`pow.cpp`](../src/ops/pow.cpp)**: Element-wise power
    - **[`sub.hpp`](ops/sub.hpp)** / **[`sub.cpp`](../src/ops/sub.cpp)**: Element-wise subtraction
    - **[`in_place.hpp`](ops/in_place.hpp)** / **[`in_place.cpp`](../src/ops/in_place.cpp)**: In-place operations
  
  - **Reduction Operations**:
    - **[`sum.hpp`](ops/sum.hpp)** / **[`sum.cpp`](../src/ops/sum.cpp)**: Sum reduction
    - **[`mean.hpp`](modules/mean.hpp)**: Mean reduction
    - **[`real_extremum.hpp`](ops/real_extremum.hpp)** / **[`real_extremum.cpp`](../src/ops/real_extremum.cpp)**: Max/min reduction
    - **[`real_unary.hpp`](ops/real_unary.hpp)** / **[`real_unary.cpp`](../src/ops/real_unary.cpp)**: Unary real operations
    - **[`reduce.hpp`](ops/reduce.hpp)** / **[`reduce.cpp`](../src/ops/reduce.cpp)**: General reduction operations
  
  - **Matrix Operations**:
    - **[`matmul.hpp`](ops/matmul.hpp)** / **[`matmul.cpp`](../src/ops/matmul.cpp)**: Matrix multiplication
    - **[`commuting.hpp`](ops/commuting.hpp)** / **[`commuting.cpp`](../src/ops/commuting.cpp)**: Commuting operations
  
  - **Broadcasting**:
    - **[`copy_broadcast.hpp`](ops/copy_broadcast.hpp)** / **[`copy_broadcast.cpp`](../src/ops/copy_broadcast.cpp)**: Broadcast copy operations
  
  - **Activation Functions**:
    - **[`softmax.hpp`](ops/softmax.hpp)** / **[`softmax.cpp`](../src/ops/softmax.cpp)**: Softmax activation
    - **[`logsoftmax.hpp`](ops/logsoftmax.hpp)** / **[`logsoftmax.cpp`](../src/ops/logsoftmax.cpp)**: Log-softmax activation
    - **[`embedding.hpp`](ops/embedding.hpp)** / **[`embedding.cpp`](../src/ops/embedding.cpp)**: Embedding operation
  
  - **Specialized Operations**:
    - **[`triu_fill.hpp`](ops/triu_fill.hpp)** / **[`triu_fill.cpp`](../src/ops/triu_fill.cpp)**: Upper triangular fill
  
  - **Utilities**:
    - **[`util.hpp`](ops/util.hpp)** / **[`util.cpp`](../src/ops/util.cpp)**: Common operation utilities

## Key Concepts

### Tensor System

The [`Tensor`](tensors/tensor.hpp) is the central data structure. It supports:
- Arbitrary dimensions with shape and stride
- Automatic differentiation via `requires_grad`
- Device placement (CPU/GPU)
- Real and complex number types
- Sparse storage for memory efficiency

### Module System

The [`Module`](modules/module.hpp) interface allows building composable neural networks:
- `forward()` method for inference
- `parameters()` method for trainable weights
- `train()`/`eval()` modes for dropout/batch norm behavior

### Storage Abstraction

The [`Storage`](storage/storage.hpp) interface abstracts memory management:
- Dense and sparse layouts
- CPU and GPU backends
- Type promotion (real → complex)

### Autograd Graph

Operations on tensors with `requires_grad=true` build a computation graph:
- [`Node`](../include/common/weed_functions.hpp) represents operations
- Backward pass computes gradients via chain rule
- Optimizers (Adam, SGD) update parameters

## Quick Start



## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0). See [`LICENSE.md`](../LICENSE.md) for details.

## Additional Files

### [`shared_api.hpp`](shared_api.hpp)
Shared C-interface library declarations.
