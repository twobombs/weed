# Weed GPU Device Abstraction

This directory contains the GPU device abstraction layer for OpenCL-based hardware acceleration.

## Overview

The `devices/` directory provides a comprehensive abstraction for managing GPU/OpenCL devices, memory allocation, and kernel dispatch. It implements a queue-based system for asynchronous kernel execution with proper event synchronization.

## Components

### [`gpu_device.hpp`](gpu_device.hpp) - Core GPU Device Manager

The `GpuDevice` struct is the central component for GPU device management:

**Key Members:**
- `deviceID`: Unique hardware device identifier
- `context`: OpenCL context for the device
- `queue`: OpenCL command queue for kernel execution
- `device_context`: Shared device context from OCLEngine singleton
- `poolItems`: Pool of pre-allocated buffer items for kernel arguments
- `wait_queue_items`: Queue of pending kernel calls
- `wait_refs`: Event dependencies for ordered execution
- `queue_mutex`: Thread synchronization for queue operations

**Key Methods:**

| Method | Description |
|--------|-------------|
| `GpuDevice(int64_t did)` | Constructor - initializes context and queue for specified device |
| `MakeBuffer(flags, size, host_ptr)` | Creates OpenCL buffer with specified flags and optional host pointer |
| `clFinish(doHard)` | Flushes and finishes the command queue; `doHard=true` flushes entire device |
| `tryOcl(message, oclCall)` | Wrapper for OpenCL calls with error handling |
| `PopQueue(isDispatch)` | Callback handler to free resources and start next queue event |
| `DispatchQueue()` | Starts the kernel dispatch and callback cycle |
| `ResetWaitEvents(waitQueue)` | Gets dependent events for next operation and clears buffer |
| `CheckCallbackError()` | Throws exception if OpenCL callback reported error |
| `AddAlloc(size)` | Tracks memory allocation; throws if VRAM limits exceeded |
| `SubtractAlloc(size)` | Decrements allocation tracking |
| `AddQueueItem(item)` | Adds kernel call to queue; dispatches if queue was empty |
| `QueueCall(api_call, wic, lgs, args, ...)` | Creates and queues a kernel call with specified parameters |
| `GetFreePoolItem()` | Retrieves unused pool item from pool |
| `RequestKernel(api_call, vciArgs, nwi, buffers, ...)` | Requests kernel execution with VCI arguments |
| `ClearIntBuffer/RealBuffer/ComplexBuffer()` | Zero-fills buffers of specified type |
| `FillOnesInt/Real/Complex()` | Fills buffers with 1.0 values |
| `FillValueInt/Real/Complex()` | Fills buffers with specified value |
| `UpcastRealBuffer()` | Converts real buffer to complex with doubled stride |
| `GetInt/Real/Complex(buffer, idx)` | Reads single element from buffer |
| `SetInt/Real/Complex(val, buffer, idx)` | Writes single element to buffer |
| `LockSync(buffer, sz, array, allow_lock)` | Maps or copies buffer to host memory |
| `UnlockSync(buffer, array)` | Unmaps previously mapped buffer |

**Memory Management:**
- Manual tracking via `AddAlloc()`/`SubtractAlloc()`
- Integration with OCLEngine's global allocation limits
- Throws `bad_alloc` with descriptive messages when limits exceeded

**Event Synchronization:**
- Maintains `wait_refs` for dependent event chains
- Ensures ordered execution across multiple kernels
- Supports both per-queue and per-device synchronization

### [`pool_item.hpp`](pool_item.hpp) - Kernel Argument Buffer Pool

The `PoolItem` struct provides pre-allocated buffers for kernel arguments:

**Purpose:**
- Reduces allocation overhead during kernel dispatch
- Reuses buffers across multiple kernel calls
- Centralizes error handling for buffer allocation

**Members:**
- `complexBuffer`: Buffer for complex arguments (size: `sizeof(complex) * CMPLX_ARG_LEN`)
- `vciBuffer`: Buffer for VCI (vector of complex index) arguments (size: `sizeof(tcapint) * VCI_ARG_LEN`)

**Key Methods:**
- `PoolItem(cl::Context &context)`: Constructor that allocates both buffers
- `MakeBuffer(context, size)`: Creates buffer with comprehensive error handling

**Error Handling:**
- Custom `bad_alloc` exception with descriptive messages
- Distinguishes between `CL_MEM_OBJECT_ALLOCATION_FAILURE`, `CL_OUT_OF_HOST_MEMORY`, `CL_INVALID_BUFFER_SIZE`

### [`queue_item.hpp`](queue_item.hpp) - Kernel Call Request Wrapper

The `QueueItem` struct wraps kernel call parameters:

**Purpose:**
- Structures kernel call parameters before pool assignment
- Maintains all necessary information for kernel execution

**Members:**
- `api_call`: OpenCL API call type (`OCLAPI`)
- `workItemCount`: Number of work items for first dimension
- `workItemCount2`: Number of work items for second dimension
- `localGroupSize`: Local work group size for first dimension
- `localGroupSize2`: Local work group size for second dimension
- `deallocSize`: Size for buffer deallocation
- `buffers`: Vector of OpenCL buffers for arguments
- `localBuffSize`: Size of local (shared) memory buffer

**Constructors:**
- Default constructor with zero-initialized members
- Full constructor with all parameters

## Design Patterns

### Singleton Pattern
The `GpuDevice` uses `WEED_GPU_SINGLETON` macro to access the `OCLEngine` singleton for device context management.

### Pool Pattern
`PoolItem` implements a pool pattern to reduce allocation overhead. `GpuDevice::GetFreePoolItem()` manages the pool lifecycle.

### Queue Pattern
Kernel calls are queued via `AddQueueItem()` and processed asynchronously. The queue ensures ordered execution with proper event synchronization.

### RAII Pattern
OpenCL resources (buffers, contexts, queues) are managed via smart pointers and RAII principles.

## Usage Example



## Thread Safety

- `queue_mutex` protects queue operations
- Each device has its own queue and mutex
- OCLEngine singleton manages global state

## Platform Considerations

- Requires OpenCL 1.1+ for `enqueueMapBuffer` support
- Local memory limits enforced via `device_context->GetLocalSize()`
- VRAM limits configurable via `WEED_MAX_OCL_MB` environment variable

## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).
