# Common Implementations

This directory contains the implementations of the common utilities and the OpenCL/CUDA kernels.

## Files

*   **weed_types.cpp**: (If present) Implementation of type-related utilities.
*   **parallel_for.cpp**: Implementation of the `ParallelFor` class, handling thread pool management (implied).
*   **oclengine.cpp**: Implementation of the OpenCL engine, handling device discovery, context creation, and kernel execution.
*   **cudaengine.cu**: Implementation of the CUDA engine (if CUDA is enabled).
*   **qengine.cl**: The main OpenCL source file containing the kernel implementations for tensor operations.
*   **qengine.cu**: The main CUDA source file containing the kernel implementations.
*   **qheader_*.cl**: Helper OpenCL headers for different data types (float, double, half, etc.).
*   **functions.cpp**: Implementation of common math functions.
*   **dispatchqueue.cpp**: Implementation of a thread-safe dispatch queue (likely used by `ParallelFor` or similar).
