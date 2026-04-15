# Weed Tensor System

This directory contains the core tensor class definitions and their specializations. The `Tensor` is the central data structure in Weed, representing multi-dimensional arrays with automatic differentiation support.

## Overview

The `tensors/` directory implements Weed's tensor system, which provides:
- Multi-dimensional arrays with arbitrary shapes
- Automatic differentiation (autograd) for gradient computation
- Device placement (CPU/GPU) for hardware acceleration
- Type support (real, complex, integer)
- Sparse tensor support for memory efficiency

## Architecture

The tensor system follows a hierarchical design:



## Base Class

### [`base_tensor.hpp`](base_tensor.hpp) - Base Tensor

Non-mathematical base class providing shape, stride, and storage access:

**Key Members**:
- `storage`: `StoragePtr` - Pointer to underlying data storage
- `shape`: `std::vector<tcapint>` - Tensor dimensions
- `stride`: `std::vector<tcapint>` - Memory strides per dimension
- `offset`: `tcapint` - Offset into storage for views

**Key Methods**:
- `copy(other)`: Shallow copy from another tensor
- `numel()`: Total number of elements
- `size(dim)`: Size of specified dimension
- `operator[](idx)`: Get sub-tensor at index
- `reshape(new_shape)`: Change tensor shape
- `transpose()`: Swap dimensions
- `flatten(axis)`: Flatten dimensions from axis

**Usage**:


## Main Tensor Class

### [`tensor.hpp`](tensor.hpp) - Tensor

Full-featured tensor with operations, autograd, and device support:

**Key Members**:
- `grad_node`: `NodePtr` - Computation graph node for backpropagation
- `grad`: `TensorPtr` - Gradient tensor
- `requires_grad`: `bool` - Whether to track gradients

**Constructors**:


**Static Factory Methods**:
| Method | Description |
|--------|-------------|
| `zeros(shape, ...)` | Create tensor filled with zeros |
| `ones_like(shape, ...)` | Create tensor filled with ones |
| `one_hot(targets, vocab_size)` | One-hot encoding |
| `make_gradient(shape, ...)` | Create gradient tensor |
| `allocate_like(orig, ...)` | Allocate without initialization |
| `contiguous(tensor)` | Ensure contiguous storage |
| `reshape(tensor, shape)` | Reshape tensor |
| `transpose(tensor, i, j)` | Swap two dimensions |
| `flatten(tensor, axis)` | Flatten dimensions |
| `chunk(tensor, n, axis)` | Split into chunks |

**Tensor Operations**:

| Operation | Method | Operator |
|-----------|--------|----------|
| Addition | `add(other)` | `+` |
| Subtraction | `sub(other)` | `-` |
| Multiplication | `mul(other)` | `*` |
| Division | `div(other)` | `/` |
| Matrix Multiply | `matmul(other)` | `>>`, `<<` |
| Power | `pow(exp)` | `^` |
| Exponential | `exp()` | - |
| Logarithm | `log()` | - |

**Activation Functions**:
| Function | Method |
|----------|--------|
| Sigmoid | `sigmoid()` |
| Tanh | `tanh()` |
| ReLU | `relu()` |
| GELU | `gelu()` |
| Softmax | `softmax(axis)` |
| Log-Softmax | `logsoftmax(axis)` |

**Reduction Operations**:
| Operation | Method |
|-----------|--------|
| Sum | `sum(axis)` |
| Mean | `mean(axis)` |
| Variance | `variance(axis)` |
| Std Dev | `stddev(axis)` |
| Max | `max(axis)` |
| Min | `min(axis)` |
| Absolute | `abs()` |

**Shape Operations**:
| Operation | Method |
|-----------|--------|
| Squeeze | `squeeze()` / `squeeze(axis)` |
| Unsqueeze | `unsqueeze(axis)` |
| Reshape | `reshape(shape)` |
| Transpose | `transpose()` / `transpose(i, j)` |
| Flatten | `flatten(axis)` |
| Slice | `slice(row)` / `slice(axis, start, length)` |

**Device Operations**:
| Operation | Method |
|-----------|--------|
| Cast to device | `cast(device)` |
| Cast in-place | `cast_in_place(device)` |

**Autograd**:
| Operation | Method |
|-----------|--------|
| Backward | `backward()` |
| Make gradient | `make_gradient(force_sparse)` |
| Reduce grad broadcast | `reduce_grad_broadcast()` |

**Operator Overloads**:


**Usage**:


## Specialized Tensors

### [`symbol_tensor.hpp`](symbol_tensor.hpp) - Symbol Tensor

Non-mathematical tensor for integer indices (e.g., token indices for embeddings):

**Key Members**:
- Inherits from `BaseTensor`
- Stores integer indices (symint type)

**Key Methods**:
- `reshape(shape)`: Change shape
- `transpose()`: Swap dimensions
- `flatten(axis)`: Flatten dimensions

**Usage**:


### [`parameter.hpp`](parameter.hpp) - Parameter

Trainable parameter wrapper:

**Key Features**:
- Inherits from `Tensor`
- `requires_grad` always true
- Used for model weights and biases

**Usage**:


## Tensor Creation Examples

### From Values


### From Scalars


### From Parameters


### Device Placement


## Autograd Flow



## Memory Layout

### Row-Major (C-style)


### Column-Major (Fortran-style)


## Broadcasting Rules

### Rules
1. Align dimensions from the right
2. Compatible if equal or one is 1
3. Result takes max of each dimension

### Examples


## Serialization

### Save/Load


### Module Serialization


## Best Practices

1. **Use `TensorPtr`**: Always use `std::shared_ptr<Tensor>` for automatic memory management
2. **Check `requires_grad`**: Only track gradients when needed to save memory
3. **Zero gradients**: Call `zero_grad()` before each training step
4. **Device consistency**: Keep tensors on same device for operations
5. **Memory efficiency**: Use in-place operations when possible
6. **Contiguous storage**: Ensure contiguous storage for efficient GPU transfers
7. **Batch operations**: Use batched operations for better GPU utilization

## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

## Additional Files

### [`complex_tensor.hpp`](complex_tensor.hpp)
Complex-valued tensor operations.

### [`flat_tensors.hpp`](flat_tensors.hpp)
Macros and operations for flattened tensors.

### [`real_tensor.hpp`](real_tensor.hpp)
Real-valued tensor operations.

### [`complex_scalar.hpp`](complex_scalar.hpp)
Complex scalar tensor type.

### [`scalar.hpp`](scalar.hpp)
Scalar base tensor type.

### [`real_scalar.hpp`](real_scalar.hpp)
Real scalar tensor type.
