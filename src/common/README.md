# Weed Common Source Code

This directory contains common utilities and OpenCL-related implementations used throughout the Weed library.

## Overview

The `src/common/` directory provides foundational functionality including:
- Mathematical and utility functions
- OpenCL engine for GPU acceleration
- Parallel execution utilities
- OpenCL kernel source files

## Files

### [`functions.cpp`](functions.cpp) - Mathematical Functions

Implementation of mathematical and utility functions used throughout the library.

**Key Functions:**



**Implementation Details:**



### [`oclengine.cpp`](oclengine.cpp) - OpenCL Engine

Implementation of the OpenCL engine for GPU-accelerated computations.

**Key Classes:**

#### `OCLContext`

Manages OpenCL context, command queue, and device.



#### `OCLKernel`

Wrapper for OpenCL kernel management.



**Kernel Compilation:**



### [`parallel_for.cpp`](parallel_for.cpp) - Parallel Execution

Implementation of parallel execution utilities using OpenMP or thread pools.

**Key Functions:**



**Implementation:**



### OpenCL Kernel Files

The following OpenCL kernel source files are included in the build:

#### [`qengine.cl`](qengine.cl) - Quantum Engine Kernel

OpenCL kernel implementation for quantum circuit simulation.



#### `qheader_*.cl` - Quantum Header Files

Header files for different data types used in quantum kernel implementations:

- `qheader_double.cl` - Double precision (real128)
- `qheader_float.cl` - Single precision (real32)
- `qheader_half.cl` - Half precision (real16)
- `qheader_quad.cl` - Quad precision (real128)
- `qheader_uint8.cl` - 8-bit unsigned integer
- `qheader_uint16.cl` - 16-bit unsigned integer
- `qheader_uint32.cl` - 32-bit unsigned integer
- `qheader_uint64.cl` - 64-bit unsigned integer

**Example Header (qheader_float.cl):**



## Usage Examples

### Creating an OpenCL Context



### Using Parallel For



## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).
