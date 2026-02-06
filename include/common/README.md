# Common Utilities

This directory contains common definitions, types, and utility classes used throughout the Weed library.

## Files

*   **weed_types.hpp**: The core type definitions for the library.
    *   `real1`: The floating-point scalar type (configurable via `FPPOW` to be `half`, `float`, `double`, or `float128`).
    *   `complex`: Complex number type based on `real1`.
    *   `tcapint`: Integer type for tensor dimensions and capacities (configurable via `TCAPPOW`).
    *   Constants: Mathematical constants (`PI_R1`, `ONE_R1`, etc.) and configuration flags.
*   **parallel_for.hpp**: Provides the `ParallelFor` class for multi-threaded execution on the CPU. It supports parallel loops over dense ranges and sparse containers.
*   **oclapi.hpp**: Defines the `OCLAPI` enum, which lists the available OpenCL kernels (e.g., `OCL_API_ADD_REAL`, `OCL_API_MATMUL_COMPLEX`).
*   **oclengine.hpp**: The OpenCL runtime manager.
    *   `OCLEngine`: A singleton that manages OpenCL devices, contexts, and program compilation.
    *   `OCLDeviceContext`: Encapsulates an OpenCL context, command queue, and device properties.
*   **weed_functions.hpp**: Declarations for common mathematical and utility functions.
*   **half.hpp**: IEEE 754-based half-precision floating-point library.
*   **serializer.hpp**: Static methods for serialization and de-serialization of primitive types (`bool`, `tcapint`, `symint`, `real1`, `complex`).
*   **config.h.in**: Template for the generated `config.h`.
