# Storage

This directory contains the classes responsible for managing the actual data of tensors. It abstracts over the device (CPU/GPU), data type (Real/Complex), and layout (Dense/Sparse).

## Files

*   **storage.hpp**: Defines the abstract `Storage` base class.
    *   Manages metadata: `DeviceTag`, `DType`, and `size`.
    *   Defines virtual interface for data movement: `cpu()`, `gpu()`.
    *   Defines utility methods: `FillZeros()`, `FillOnes()`, `Upcast()`.
*   **all_storage.hpp**: Convenience header including all storage types.
*   **real_storage.hpp**, **complex_storage.hpp**: Intermediate interfaces for Real and Complex data types, defining virtual methods for element access (`operator[]`, `write`, `add`).
*   **cpu_real_storage.hpp**, **cpu_complex_storage.hpp**: Dense implementations for CPU.
*   **sparse_cpu_real_storage.hpp**, **sparse_cpu_complex_storage.hpp**: Sparse implementations for CPU, using hash maps (`std::unordered_map`) to store only non-zero elements.
*   **gpu_storage.hpp**: Interface for GPU storage.
*   **gpu_real_storage.hpp**, **gpu_complex_storage.hpp**: Dense implementations for GPU, managing OpenCL/CUDA buffers.
