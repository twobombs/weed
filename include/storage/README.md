# Weed Storage Abstraction

This directory contains the storage abstraction layer for tensor data management, supporting both CPU and GPU backends with dense and sparse layouts.

## Overview

The `storage/` directory implements a flexible storage abstraction that separates data layout from tensor operations. This design enables:
- Multiple backends (CPU, GPU/OpenCL)
- Different layouts (dense, sparse)
- Type specialization (real, complex, integer)
- Efficient memory management
- Device migration

## Base Class

### [`storage.hpp`](storage.hpp) - Storage Interface

The `Storage` class is the abstract base for all storage implementations:

**Key Members:**
- `stype`: Storage type identifier
- `device`: Device tag (CPU/GPU)
- `dtype`: Data type (REAL/COMPLEX/INT)
- `size`: Number of elements

**Key Methods:**
- `FillZeros()`: Pure virtual - fill with zeros
- `FillOnes()`: Pure virtual - fill with ones
- `Upcast(dtype)`: Pure virtual - convert to different type
- `is_sparse()`: Returns true for sparse storage
- `get_sparse_size()`: Returns sparse element count
- `is_gpu()`: Returns true for GPU storage
- `cpu()`: Pure virtual - migrate to CPU
- `gpu(device_id)`: Pure virtual - migrate to GPU
- `save(std::ostream&)`: Serialize to stream
- `load(std::istream&)`: Static factory for deserialization

**Serialization:**


## Typed Storage

### [`typed_storage.hpp`](typed_storage.hpp) - Template Storage

Template-based storage implementation:

**Template Parameter:**
- `T`: Element type (`real1`, `complex`, or `tcapint`)

**Key Methods:**
- `operator[](idx)`: Element access (pure virtual)
- `write(idx, val)`: Write element (pure virtual)
- `add(idx, val)`: Add to element (pure virtual)
- `FillValue(val)`: Fill with value
- `Upcast(dtype)`: Pure virtual - type conversion

**Type Aliases:**
- `IntStorage`: `TypedStorage<tcapint>`
- `RealStorage`: `TypedStorage<real1>`
- `ComplexStorage`: `TypedStorage<complex>`

**Memory Allocation:**
- Aligned allocation (64-byte alignment)
- Platform-specific optimizations:
  - Windows: `_aligned_malloc`/`_aligned_free`
  - macOS: `posix_memalign`
  - Linux: `aligned_alloc`

## CPU Storage

### [`cpu_storage.hpp`](cpu_storage.hpp) - CPU Storage Base

Base class for CPU storage implementations.

### [`cpu_real_storage.hpp`](cpu_real_storage.hpp) - Real CPU Storage

**Features:**
- Dense real-valued storage on CPU
- Aligned memory allocation
- Fast element access

**Implementation:**


### [`cpu_complex_storage.hpp`](cpu_complex_storage.hpp) - Complex CPU Storage

**Features:**
- Dense complex-valued storage on CPU
- Real and imaginary components
- Complex arithmetic support

### [`cpu_int_storage.hpp`](cpu_int_storage.hpp) - Integer CPU Storage

**Features:**
- Integer storage for SymbolTensor
- Used for token indices, embeddings
- No arithmetic operations

## GPU Storage

### [`gpu_storage.hpp`](gpu_storage.hpp) - GPU Storage Base

Base class for GPU storage implementations:

**Key Methods:**
- `is_gpu()`: Returns true
- `cpu()`: Migrate to CPU via `enqueueReadBuffer`
- `gpu(device_id)`: Returns self or migrates

### [`gpu_real_storage.hpp`](gpu_real_storage.hpp) - Real GPU Storage

**Features:**
- OpenCL buffer-backed storage
- Asynchronous data transfer
- Zero-copy support with host pointers

**Implementation:**


### [`gpu_complex_storage.hpp`](gpu_complex_storage.hpp) - Complex GPU Storage

**Features:**
- OpenCL buffer for complex values
- Complex arithmetic on GPU

### [`gpu_int_storage.hpp`](gpu_int_storage.hpp) - Integer GPU Storage

**Features:**
- GPU integer storage
- Used for GPU-based indexing

## Sparse Storage

### [`sparse_cpu_storage.hpp`](sparse_cpu_storage.hpp) - Sparse CPU Storage Base

Base class for sparse CPU storage:

**Features:**
- Hash-based sparse representation
- `std::unordered_map<tcapint, T>` storage
- Memory efficient for sparse data

### [`sparse_cpu_real_storage.hpp`](sparse_cpu_real_storage.hpp) - Real Sparse Storage

**Implementation:**


### [`sparse_cpu_complex_storage.hpp`](sparse_cpu_complex_storage.hpp) - Complex Sparse Storage

**Features:**
- Sparse complex storage
- Hash-based lookup
- Zero for missing indices

## Storage Factory

### [`all_storage.hpp`](all_storage.hpp) - Storage Factory

Factory function for creating storage:



**Factory Logic:**


## Device Migration

### CPU to GPU



### GPU to CPU



## Usage Example



## Performance Considerations

1. **Alignment**: 64-byte alignment for SIMD optimization
2. **Batching**: GPU transfers should be batched
3. **Reuse**: Pool items reduce allocation overhead
4. **Sparse**: Use sparse storage for >90% zero data
5. **Contiguity**: Ensure contiguous storage for efficient transfers

## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).
