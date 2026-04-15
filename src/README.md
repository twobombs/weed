# Weed Source Code

This directory contains the implementation files for the Weed neural network library. The code is organized into subdirectories corresponding to the header files in the `include/` directory.

## Overview

The `src/` directory implements the functionality declared in the `include/` directory. Each subdirectory mirrors the structure of the include directory:

- `common/` - Common utilities and OpenCL kernel implementations
- `devices/` - Device-specific implementations (GPU)
- `modules/` - Neural network layer implementations
- `ops/` - Tensor operation implementations
- `storage/` - Storage backend implementations (CPU, GPU)
- `tensors/` - Tensor class implementations

## File Structure

### [`src/README.md`](README.md) - Source Directory Overview

This file provides an overview of the source code structure.

### [`src/common/`](common/) - Common Utilities

**Files:**

- [`functions.cpp`](common/functions.cpp) - Mathematical and utility functions
- [`oclengine.cpp`](common/oclengine.cpp) - OpenCL engine implementation
- [`parallel_for.cpp`](common/parallel_for.cpp) - Parallel execution utilities
- `qengine.cl` - OpenCL kernel for quantum engine
- `qheader_*.cl` - OpenCL header files for different data types

**Key Implementations:**

#### [`functions.cpp`](common/functions.cpp)



#### [`oclengine.cpp`](common/oclengine.cpp)



#### [`parallel_for.cpp`](common/parallel_for.cpp)



### [`src/devices/`](devices/) - Device Implementations

**Files:**

- [`gpu_device.cpp`](devices/gpu_device.cpp) - GPU device implementation

**Key Implementations:**

#### [`gpu_device.cpp`](devices/gpu_device.cpp)



### [`src/modules/`](modules/) - Neural Network Layers

**Files:**

- [`module.cpp`](modules/module.cpp) - Base module implementation
- [`linear.cpp`](modules/linear.cpp) - Linear layer
- [`sequential.cpp`](modules/sequential.cpp) - Sequential container
- [`dropout.cpp`](modules/dropout.cpp) - Dropout layer
- [`layernorm.cpp`](modules/layernorm.cpp) - Layer normalization
- `RMSNorm` - RMS normalization
- [`embedding.cpp`](modules/embedding.cpp) - Embedding layer
- [`gru.cpp`](modules/gru.cpp) - GRU layer
- [`lstm.cpp`](modules/lstm.cpp) - LSTM layer
- [`positional_encoding.cpp`](modules/positional_encoding.cpp) - Positional encoding
- [`multihead_attention.cpp`](modules/multihead_attention.cpp) - Multi-head attention
- [`transformer_encoder_layer.cpp`](modules/transformer_encoder_layer.cpp) - Transformer encoder
- [`qrack_neuron.cpp`](modules/qrack_neuron.cpp) - Quantum neuron
- [`qrack_neuron_layer.cpp`](modules/qrack_neuron_layer.cpp) - Quantum neuron layer
- [`migrate_cpu.cpp`](modules/migrate_cpu.cpp) - CPU migration
- [`migrate_gpu.cpp`](modules/migrate_gpu.cpp) - GPU migration
- [`learned_positional_encoding.cpp`](modules/learned_positional_encoding.cpp) - Learned positional encoding
- [`qwen.cpp`](modules/qwen.cpp) - Qwen model
- [`qwen_decoder_layer.cpp`](modules/qwen_decoder_layer.cpp) - Qwen decoder layer
- [`qwen_model.cpp`](modules/qwen_model.cpp) - Qwen complete model (Token Embedding, Stack of QwenDecoderLayer, Final RMSNorm, LM Head)
- [`qwen_tokenizer.cpp`](modules/qwen_tokenizer.cpp) - Qwen byte-level BPE tokenizer
- [`rope.cpp`](modules/rope.cpp) - Rotary positional encoding

**Key Implementations:**

#### [`module.cpp`](modules/module.cpp)



#### [`linear.cpp`](modules/linear.cpp)



#### [`sequential.cpp`](modules/sequential.cpp)



#### [`dropout.cpp`](modules/dropout.cpp)



#### [`layernorm.cpp`](modules/layernorm.cpp)



#### [`multihead_attention.cpp`](modules/multihead_attention.cpp)



#### [`transformer_encoder_layer.cpp`](modules/transformer_encoder_layer.cpp)



### [`src/ops/`](ops/) - Tensor Operations

**Files:**

- [`util.cpp`](ops/util.cpp) - Utility functions for tensor operations
- [`sum.cpp`](ops/sum.cpp) - Sum reduction
- [`abs.cpp`](ops/abs.cpp) - Absolute value
- [`clamp.cpp`](ops/clamp.cpp) - Clamping values
- [`commuting.cpp`](ops/commuting.cpp) - Commuting operations
- [`copy_broadcast.cpp`](ops/copy_broadcast.cpp) - Copy and broadcast
- [`div.cpp`](ops/div.cpp) - Division
- [`embedding.cpp`](ops/embedding.cpp) - Embedding lookup
- [`in_place.cpp`](ops/in_place.cpp) - In-place operations
- [`logsoftmax.cpp`](ops/logsoftmax.cpp) - Log-softmax
- [`matmul.cpp`](ops/matmul.cpp) - Matrix multiplication
- [`pow.cpp`](ops/pow.cpp) - Power operation
- [`real_extremum.cpp`](ops/real_extremum.cpp) - Real extremum operations
- [`real_unary.cpp`](ops/real_unary.cpp) - Real unary operations
- [`reduce.cpp`](ops/reduce.cpp) - Reduction operations
- [`softmax.cpp`](ops/softmax.cpp) - Softmax
- [`sub.cpp`](ops/sub.cpp) - Subtraction
- [`triu_fill.cpp`](ops/triu_fill.cpp) - Upper triangular fill

**Key Implementations:**

#### [`util.cpp`](ops/util.cpp)



#### [`matmul.cpp`](ops/matmul.cpp)



#### [`softmax.cpp`](ops/softmax.cpp)



### [`src/storage/`](storage/) - Storage Backends

**Files:**

- [`storage.cpp`](storage/storage.cpp) - Base storage class
- `CpuStorage` - CPU storage implementation
- [`cpu_real_storage.cpp`](storage/cpu_real_storage.cpp) - CPU real storage
- [`cpu_int_storage.cpp`](storage/cpu_int_storage.cpp) - CPU int storage
- [`cpu_complex_storage.cpp`](storage/cpu_complex_storage.cpp) - CPU complex storage
- `GpuStorage` - GPU storage base
- [`gpu_real_storage.cpp`](storage/gpu_real_storage.cpp) - GPU real storage
- [`gpu_int_storage.cpp`](storage/gpu_int_storage.cpp) - GPU int storage
- [`gpu_complex_storage.cpp`](storage/gpu_complex_storage.cpp) - GPU complex storage
- `SparseCpuStorage` - Sparse CPU storage
- [`sparse_cpu_real_storage.cpp`](storage/sparse_cpu_real_storage.cpp) - Sparse CPU real storage
- [`sparse_cpu_complex_storage.cpp`](storage/sparse_cpu_complex_storage.cpp) - Sparse CPU complex storage

**Key Implementations:**

#### [`storage.cpp`](storage/storage.cpp)



#### [`cpu_real_storage.cpp`](storage/cpu_real_storage.cpp)



#### [`gpu_real_storage.cpp`](storage/gpu_real_storage.cpp)



### [`src/tensors/`](tensors/) - Tensor Implementations

**Files:**

- [`tensor.cpp`](tensors/tensor.cpp) - Base tensor implementation
- [`base_tensor.cpp`](tensors/base_tensor.cpp) - Base tensor class
- [`parameter.cpp`](tensors/parameter.cpp) - Parameter (trainable tensor)
- [`symbol_tensor.cpp`](tensors/symbol_tensor.cpp) - Symbolic tensor

**Key Implementations:**

#### [`tensor.cpp`](tensors/tensor.cpp)



#### [`parameter.cpp`](tensors/parameter.cpp)



## Build System

The project uses CMake for building. Key build files:

- [`CMakeLists.txt`](../CMakeLists.txt) - Main CMake configuration
- [`cmake/`](../cmake/) - CMake modules and toolchains

### Build Targets



## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

## Additional Files

### [`shared_api.cpp`](shared_api.cpp)
Shared C-interface library API wrapper.

### [`weed_cl_precompile.cpp`](weed_cl_precompile.cpp)
OpenCL kernel precompilation build tool.
