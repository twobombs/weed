# Source Code

This directory contains the implementation of the Weed library.

## Subdirectories

*   **common/**: Implementations of common utilities and OpenCL/CUDA kernels.
*   **devices/**: Implementations of device management.
*   **modules/**: Implementations of neural network modules.
*   **ops/**: Implementations of tensor operation dispatch logic.
*   **storage/**: Implementations of storage classes (CPU/GPU, Sparse/Dense).
*   **tensors/**: Implementations of the `Tensor` logic.

## Files

*   **weed_cl_precompile.cpp**: A utility program that precompiles OpenCL kernels and saves them to disk to speed up subsequent load times.
