# Common Implementations

This directory contains the implementations of the common utilities and the OpenCL/CUDA kernels.

## Files

*   **parallel_for.cpp**: Implementation of the `ParallelFor` class, handling thread pool management (implied).
*   **oclengine.cpp**: Implementation of the OpenCL engine, handling device discovery, context creation, and kernel execution.
*   **qengine.cl**: The main OpenCL source file containing the kernel implementations for tensor operations.
*   **qheader_*.cl**: Helper OpenCL headers for different data types (float, double, half, etc.).
*   **functions.cpp**: Implementation of common math functions.
