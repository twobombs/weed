# Common Utilities

This directory contains fundamental type definitions, utilities, and infrastructure components used throughout the Weed library. These are the building blocks upon which tensors, modules, and operations are constructed.

## Files

### [`weed_types.hpp`](weed_types.hpp)
Core type definitions and constants for the library.

#### Type Definitions
| Type | Description |
|------|-------------|
| `real1` | Primary floating-point scalar type (configurable via `WEED_FPPOW`) |
| `complex` | Complex number type: `std::complex<real1>` |
| `tcapint` | Tensor capacity integer for dimensions/capacities |
| `symint` | Symbolic integer for embedding indices |
| `tlenint` | Length integer for iteration |
| `RealPtr` | `std::unique_ptr<real1[]>` for memory management |
| `ComplexPtr` | `std::unique_ptr<complex[]>` for memory management |
| `RealSparseVector` | `std::unordered_map<tcapint, real1>` for sparse data |
| `ComplexSparseVector` | `std::unordered_map<tcapint, complex>` for sparse data |

#### Configuration Macros
| Macro | Description |
|-------|-------------|
| `WEED_FPPOW` | Floating-point precision power (4=half, 5=float, 6=double, 7=float128) |
| `WEED_TCAPPOW` | Tensor capacity integer power (3-7 for 8-128 bit) |
| `WEED_CONST` | `constexpr` or `const` based on precision |

#### Mathematical Constants
| Constant | Value |
|----------|-------|
| `PI_R1` | π (pi) |
| `SQRT2_R1` | √2 |
| `SQRT1_2_R1` | 1/√2 |
| `E_R1` | Euler's number e |
| `ONE_R1` | 1.0 |
| `HALF_R1` | 0.5 |
| `ZERO_R1` | 0.0 |
| `ONE_CMPLX` | (1, 0) complex |
| `ZERO_CMPLX` | (0, 0) complex |
| `I_CMPLX` | (0, 1) complex |

#### Epsilon Values
| Constant | Description |
|----------|-------------|
| `REAL1_EPSILON` | Half the probability in any single permutation of maximally superposed qubits |
| `FP_NORM_EPSILON` | Floating-point norm epsilon for comparisons |

### [`parallel_for.hpp`](parallel_for.hpp)
Multi-threaded parallel execution utilities for CPU operations.

#### Class: `ParallelFor`
Manages parallel loop execution across multiple CPU cores.

| Method | Description |
|--------|-------------|
| `GetNumCores()` | Returns the number of available CPU cores |
| `SetConcurrencyLevel(unsigned)` | Sets the target concurrency level |
| `GetStride()` | Returns the parallelization stride |
| `par_for(begin, end, fn)` | Parallel loop over dense range |
| `par_for_inc(begin, count, inc, fn)` | Parallel loop with custom increment |
| `par_for(sparseMap, fn)` | Parallel iteration over sparse containers |

#### Usage Example


### [`oclapi.hpp`](oclapi.hpp)
OpenCL API enumeration defining all available kernels.

#### Enum: `OCLAPI`
Lists all OpenCL kernel operations supported by the library.

| Category | API Values |
|----------|------------|
| **Real Operations** | `OCL_API_ADD_REAL`, `OCL_API_SUB_REAL`, `OCL_API_MUL_REAL`, `OCL_API_DIV_REAL` |
| **Complex Operations** | `OCL_API_ADD_COMPLEX`, `OCL_API_SUB_COMPLEX`, `OCL_API_MUL_COMPLEX`, `OCL_API_DIV_COMPLEX` |
| **Matrix Operations** | `OCL_API_MATMUL_REAL`, `OCL_API_MATMUL_COMPLEX` |
| **Activation Functions** | `OCL_API_RELU`, `OCL_API_SIGMOID`, `OCL_API_TANH`, `OCL_API_GELU` |
| **Reduction** | `OCL_API_SUM_REAL`, `OCL_API_SUM_COMPLEX`, `OCL_API_MEAN_REAL` |
| **Memory** | `OCL_API_FILL_ZERO`, `OCL_API_FILL_ONE`, `OCL_API_COPY` |

### [`oclengine.hpp`](oclengine.hpp)
OpenCL runtime manager for GPU acceleration.

#### Class: `OCLDeviceContext`
Encapsulates a single OpenCL device context.

| Member | Description |
|--------|-------------|
| `platform` | OpenCL platform |
| `device` | OpenCL device |
| `context` | OpenCL context |
| `queue` | Command queue for kernel execution |
| `calls` | Map of compiled kernels by `OCLAPI` |

| Method | Description |
|--------|-------------|
| `GetKernel(OCLAPI)` | Get or compile a kernel for the given API |
| `EnqueueCall()` | Execute a queued kernel call |

#### Class: `OCLEngine` (Singleton)
Manages all OpenCL devices and contexts.

| Method | Description |
|--------|-------------|
| `GetDeviceContext(DeviceTag, id)` | Get or create a device context |
| `GetNumDevices()` | Returns number of available devices |
| `GetDevice()` | Get the default device |

### [`weed_functions.hpp`](weed_functions.hpp)
Declarations for common mathematical and utility functions.

| Function | Description |
|----------|-------------|
| `exp()`, `log()`, `sqrt()` | Basic math functions |
| `relu()`, `sigmoid()`, `tanh()` | Activation functions |
| `gelu()`, `gelu_backward()` | GELU activation |
| `softmax()`, `log_softmax()` | Softmax operations |
| `random_normal()`, `random_uniform()` | Random number generation |

### [`half.hpp`](half.hpp)
IEEE 754-based half-precision floating-point library.

Provides:
- `half_float::half` type for 16-bit floating-point
- Arithmetic operators and conversions
- Memory-efficient storage for neural network weights

### [`serializer.hpp`](serializer.hpp)
Serialization utilities for saving and loading models.

| Function | Description |
|----------|-------------|
| `serialize_bool()`, `serialize_tcapint()` | Serialize primitive types |
| `serialize_real1()`, `serialize_complex()` | Serialize floating-point types |
| `de_serialize_*()` | Reverse operations for loading |

### [`config.h.in`](config.h.in)
CMake template for generating `config.h` with build-time configuration.

## Dependencies

- **OpenCL**: For GPU acceleration (optional, controlled by `WEED_ENABLE_OPENCL`)
- **Boost**: For float128 support (optional, when `WEED_FPPOW >= 7`)
- **stdfloat** (C++23): For native half-precision support

## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

## Additional Files

### [`rapidcsv.h`](rapidcsv.h)
Third-party single-header CSV parser library.
