# Enumerations

This directory contains enumeration definitions used throughout the Weed library.

## Files

*   **device_tag.hpp**: Defines the `DeviceTag` enum, specifying the backend device type for tensors.
    *   `CPU`: CPU-based storage and execution.
    *   `GPU`: GPU-based storage and execution (OpenCL/CUDA).
    *   `Qrack`: (Experimental/Future) Qrack-based execution.
*   **dtype.hpp**: Defines the `DType` enum, specifying the data type of the tensor elements.
    *   `REAL`: Real numbers (precision defined by `real1`).
    *   `COMPLEX`: Complex numbers.
