# Tensors

This directory contains the core tensor class definitions and their specializations.

## Files

*   **base_tensor.hpp**: Defines the `BaseTensor` struct, a non-mathematical base tensor for indexing (shape, stride, storage pointer).
*   **tensor.hpp**: Defines the `Tensor` struct, the central data structure in Weed (inherits from `BaseTensor`).
    *   **Properties**:
        *   `storage`: Pointer to the underlying data storage (`StoragePtr`).
        *   `shape` & `stride`: Vectors defining the tensor's dimensions and memory layout.
        *   `grad`: Gradient tensor (if `requires_grad` is true).
        *   `grad_node`: Pointer to the node in the autograd computation graph.
    *   **Operations**: Static methods for math operations (`add`, `matmul`, `relu`, etc.) that execute the operation and build the computation graph.
    *   **Operators**: Overloads for `+`, `-`, `*`, `/`, `>>` (matmul), `^` (pow).
*   **symbol_tensor.hpp**: Defines `SymbolTensor`, a non-mathematical tensor solely for indexing (by integer enumeration), e.g., for embeddings.
*   **parameter.hpp**: Defines `Parameter`, a subclass of `Tensor` that defaults to `requires_grad=true`. Used for learnable weights in `Module`s.
*   **scalar.hpp**: Defines `Scalar`, a subclass of `Tensor` representing a single value (rank-0 tensor).
*   **real_tensor.hpp**, **complex_tensor.hpp**: (If used directly) Specializations for real and complex tensors.
*   **real_scalar.hpp**, **complex_scalar.hpp**: Specializations for real and complex scalars.
*   **flat_tensors.hpp**: Utilities for flattening tensors (likely for serialization or specific operations).
