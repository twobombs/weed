# Storage

This directory contains the classes responsible for managing the actual data of tensors. It abstracts over the device (CPU/GPU), data type (Real/Complex), and layout (Dense/Sparse).

## Files

*   **storage.hpp**: Defines the abstract `Storage` base class.
    *   Manages metadata: `DeviceTag`, `DType`, and `size`.
    *   Defines virtual interface for data movement: `cpu()`, `gpu()`.
    *   Defines utility methods: `FillZeros()`, `FillOnes()`, `Upcast()`.
*   **all_storage.hpp**: Convenience header including all storage types.
*   **typed_storage.hpp**: Template base class `TypedStorage<T>` implementing type-specific logic.
*   **cpu_storage.hpp**: Base template `CpuStorage<T>` for CPU-accessible storage.
*   **sparse_cpu_storage.hpp**: Base template `SparseCpuStorage<T>` for sparse CPU storage using hash maps.
*   **cpu_real_storage.hpp**, **cpu_complex_storage.hpp**, **cpu_int_storage.hpp**: Dense implementations for CPU.
*   **sparse_cpu_real_storage.hpp**, **sparse_cpu_complex_storage.hpp**: Sparse implementations for CPU.
*   **gpu_storage.hpp**: Interface for GPU storage.
*   **gpu_real_storage.hpp**, **gpu_complex_storage.hpp**, **gpu_int_storage.hpp**: Dense implementations for GPU, managing OpenCL/CUDA buffers.
