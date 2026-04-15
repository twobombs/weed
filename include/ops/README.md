# Weed Tensor Operations

This directory contains the implementation of tensor operations with automatic differentiation support. Operations are implemented as kernel structures that abstract execution across different devices (CPU/GPU) and data types (Real/Complex).

## Overview

The `ops/` directory implements the core tensor operations that power Weed's computational engine. Each operation:
- Supports both CPU and GPU execution via OpenCL kernels
- Handles real and complex number types
- Implements autograd nodes for backpropagation
- Supports broadcasting for shape compatibility
- Provides operator overloading for natural syntax

## Architecture

### Operation Pattern

Each operation follows a consistent pattern:

1. **Kernel Structure**: Defines the operation for specific types
2. **Dispatch Function**: Routes to appropriate kernel based on device/type
3. **Autograd Node**: Builds computation graph for backpropagation
4. **Broadcasting**: Handles shape compatibility

### Kernel Types

| Kernel Type | Description |
|-------------|-------------|
| `CommutingKernel` | Base for commutative operations (add, mul) |
| `InPlaceKernel` | Base for in-place operations |
| `ReduceKernel` | Base for reduction operations |
| `UnaryKernel` | Base for unary operations |
| `BinaryKernel` | Base for binary operations |

## Element-wise Operations

### [`abs.hpp`](abs.hpp) / [`abs.cpp`](../src/ops/abs.cpp) - Absolute Value

Computes element-wise absolute value: `|x|`

**Kernel**: `AbsKernel`

**Autograd**:
- Forward: `y = |x|`
- Backward: `dL/dx = sign(x) * dL/dy`

**Usage**:


### [`clamp.hpp`](clamp.hpp) / [`clamp.cpp`](clamp.cpp) - Value Clamping

Clamps values to range: `clamp(x, min, max)`

**Kernel**: `ClampKernel`

**Autograd**:
- Forward: `y = max(min, min(max, x))`
- Backward: `dL/dx = dL/dy` where `min <= x <= max`, else 0

**Usage**:


### [`div.hpp`](div.hpp) / [`div.cpp`](div.cpp) - Division

Element-wise division: `a / b`

**Kernel**: `DivKernel`

**Autograd**:
- Forward: `y = a / b`
- Backward: `dL/da = db/dy / b`, `dL/db = -a * dL/da / b^2`

**Usage**:


**Operator Overload**:


### [`pow.hpp`](pow.hpp) / [`pow.cpp`](pow.cpp) - Power, Exp, Log

Power operation: `a ^ b`

**Kernel**: `PowKernel`

**Autograd**:
- Forward: `y = a^b`
- Backward: `dL/da = b * a^(b-1) * dL/dy`, `dL/db = a^b * ln(a) * dL/dy`

**Usage**:


**Exponential**:


**Logarithm**:


**Operator Overload**:


### [`sub.hpp`](sub.hpp) / [`sub.cpp`](../src/ops/sub.cpp) - Subtraction

Element-wise subtraction: `a - b`

**Kernel**: `SubKernel`

**Autograd**:
- Forward: `y = a - b`
- Backward: `dL/da = dL/dy`, `dL/db = -dL/dy`

**Usage**:


**Operator Overload**:


### [`in_place.hpp`](in_place.hpp) / [`in_place.cpp`](../src/ops/in_place.cpp) - In-place Operations

In-place modification operations:

**Functions**:
- `add_in_place(a, b)`: `a = a + b`
- `sub_in_place(a, b)`: `a = a - b`
- `mul_in_place(a, b)`: `a = a * b`
- `div_in_place(a, b)`: `a = a / b`

**Usage**:


## Reduction Operations

### [`sum.hpp`](sum.hpp) / [`sum.cpp`](../src/ops/sum.cpp) - Sum and Mean

Sum reduction over dimension: `sum(x, dim)`

**Kernel**: `SumKernel`

**Autograd**:
- Forward: `y = sum(x, dim)`
- Backward: `dL/dx = broadcast(dL/dy, dim)`

**Usage**:


### [`real_extremum.hpp`](real_extremum.hpp) / [`real_extremum.cpp`](../src/ops/real_extremum.cpp) - Max/Min

Maximum and minimum over dimension: `max(x, dim)`, `min(x, dim)`

**Kernel**: `RealExtremumKernel`

**Autograd**:
- Forward: `y = max(x, dim)`
- Backward: `dL/dx = dL/dy` at max/min indices, else 0

**Usage**:


### [`reduce.hpp`](reduce.hpp) / [`reduce.cpp`](../src/ops/reduce.cpp) - General Reduction

Base reduction kernel for custom reductions.

### [`variance.hpp`](../modules/variance.hpp) - Variance/StdDev

Variance and standard deviation: `var(x, dim)`, `std(x, dim)`

**Autograd**:
- Forward: `y = var(x, dim)`
- Backward: `dL/dx = 2 * mean(x - mean(x)) * dL/dy`

**Usage**:


## Matrix Operations

### [`matmul.hpp`](matmul.hpp) / [`matmul.cpp`](../src/ops/matmul.cpp) - Matrix Multiplication

Matrix multiplication: `A @ B`

**Kernel**: `MatMulKernel`

**Autograd**:
- Forward: `Y = A @ B`
- Backward: `dL/dA = dL/dY @ B^T`, `dL/dB = A^T @ dL/dY`

**Usage**:


**Operator Overload**:


### [`commuting.hpp`](commuting.hpp) / [`commuting.cpp`](../src/ops/commuting.cpp) - Commutative Operations

Base for commutative binary operations:

**Operations**:
- Addition: `add(a, b)`
- Multiplication: `mul(a, b)`

**Autograd**:
- Add: `dL/da = dL/dy`, `dL/db = dL/dy`
- Mul: `dL/da = b * dL/dy`, `dL/db = a * dL/dy`

**Usage**:


**Operator Overload**:


## Activation Functions

### [`softmax.hpp`](softmax.hpp) / [`softmax.cpp`](../src/ops/softmax.cpp) - Softmax

Softmax activation: `softmax(x, dim) = exp(x) / sum(exp(x))`

**Kernel**: `SoftmaxKernel`

**Autograd**:
- Forward: `y_i = exp(x_i) / sum_j(exp(x_j))`
- Backward: Jacobian-vector product computed efficiently

**Usage**:


### [`logsoftmax.hpp`](logsoftmax.hpp) / [`logsoftmax.cpp`](../src/ops/logsoftmax.cpp) - Log-Softmax

Log-softmax activation: `log_softmax(x, dim) = x - log(sum(exp(x)))`

**Kernel**: `LogSoftmaxKernel`

**Autograd**:
- Forward: `y_i = x_i - log(sum_j(exp(x_j)))`
- Backward: Efficient Jacobian computation

**Usage**:


### [`embedding.hpp`](embedding.hpp) / [`embedding.cpp`](../src/ops/embedding.cpp) - Embedding Lookup

Embedding lookup operation: `lookup(embeddings, indices)`

**Kernel**: `EmbeddingKernel`

**Autograd**:
- Forward: `y = embeddings[indices]`
- Backward: `dL/dembeddings = scatter_add(dL/y, indices)`

**Usage**:


## Specialized Operations

### [`triu_fill.hpp`](triu_fill.hpp) / [`triu_fill.cpp`](../src/ops/triu_fill.cpp) - Upper Triangular Fill

Fill upper triangular part of matrix: `triu_fill(x, diagonal)`

**Kernel**: `TriuFillKernel`

**Usage**:


### [`copy_broadcast.hpp`](copy_broadcast.hpp) / [`copy_broadcast.cpp`](../src/ops/copy_broadcast.cpp) - Broadcast Copy

Materialize broadcast indices for shape compatibility.

**Kernel**: `CopyBroadcastKernel`

**Usage**:


## Utilities

### [`util.hpp`](util.hpp) / [`util.cpp`](../src/ops/util.cpp) - Operation Utilities

Helper functions for operation validation:

**Functions**:
- `SameDevice(a, b)`: Check if tensors on same device
- `SameShape(a, b)`: Check if tensors have same shape
- `ValidateShape(shape1, shape2)`: Validate shapes for operation
- `is_contiguous(shape, stride)`: Check if storage is contiguous
- `full_contiguous_stride(shape)`: Compute contiguous strides
- `broadcast_shape(shape1, shape2)`: Compute broadcast shape

**Usage**:


## Broadcasting Rules

Weed follows NumPy-style broadcasting:

1. **Align from right**: Dimensions aligned from the last axis
2. **Compatibility**: Dimensions compatible if equal or one is 1
3. **Result shape**: Max of each dimension

**Examples**:


## Type Promotion

| Operation | Result Type |
|-----------|-------------|
| Real + Real | Real |
| Complex + Complex | Complex |
| Real + Complex | Complex |
| Real * Real | Real |
| Complex * Complex | Complex |
| Real * Complex | Complex |

## Device Dispatch

Operations automatically dispatch based on tensor device:



## Autograd Integration

All operations build computation graph nodes when `requires_grad=true`:



## Performance Considerations

1. **Kernel Fusion**: Multiple operations can be fused on GPU
2. **Memory Coalescing**: GPU kernels use coalesced memory access
3. **Local Memory**: Shared memory for reduction operations
4. **Batching**: Batch operations for efficiency
5. **In-place**: Use in-place operations to save memory

## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

## Additional Files

### [`real_unary.hpp`](real_unary.hpp)
Unary kernel operations for real data types.
