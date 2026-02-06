# Storage Implementations

This directory contains the implementations of the storage classes.

## Files

*   **storage.cpp**: Implementation of base `Storage` methods.
*   **cpu_real_storage.cpp**, **cpu_complex_storage.cpp**, **cpu_int_storage.cpp**: Implementations for dense CPU storage.
    *   Handles memory allocation (`Alloc`) and deallocation.
    *   Implements data transfer logic (to/from GPU).
*   **sparse_cpu_real_storage.cpp**, **sparse_cpu_complex_storage.cpp**: Implementations for sparse CPU storage.
*   **gpu_real_storage.cpp**, **gpu_complex_storage.cpp**, **gpu_int_storage.cpp**: Implementations for GPU storage.
    *   Interacts with `GpuDevice` to manage device buffers.
    *   Implements data transfer logic (to/from CPU).
    *   Uses kernels for operations like `FillZeros`.
