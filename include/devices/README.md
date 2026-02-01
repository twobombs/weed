# Device Abstraction

This directory contains classes for managing hardware devices, specifically GPUs (via OpenCL or CUDA).

## Files

*   **gpu_device.hpp**: Defines the `GpuDevice` class.
    *   `GpuDevice`: The primary interface for interacting with a GPU. It manages the OpenCL/CUDA context, command queue, and memory allocations. It provides high-level methods to execute kernels (e.g., `FillValueReal`, `DispatchQueue`) and manage buffers (`MakeBuffer`, `LockSync`, `UnlockSync`).
*   **pool_item.hpp**: Defines `PoolItem`, which manages reusable, pre-allocated buffers for passing scalar arguments (like complex numbers or dimensions) to kernels, reducing allocation overhead.
*   **queue_item.hpp**: Defines `QueueItem`, a simple struct that encapsulates the parameters for a pending kernel execution request (API call ID, work sizes, buffers) before it is processed by the device queue.
