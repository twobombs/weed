# Device Implementations

This directory contains the implementation of the device management classes.

## Files

*   **gpu_device.cpp**: Implements the `GpuDevice` class. It handles the details of:
    *   Creating and managing OpenCL/CUDA buffers.
    *   Enqueuing kernel execution commands to the command queue.
    *   Synchronizing memory access between host and device (`LockSync`, `UnlockSync`).
    *   Dispatching specific operations (like filling buffers or copying data) to the underlying engine.
