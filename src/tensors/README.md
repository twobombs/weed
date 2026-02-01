# Tensor Implementations

This directory contains the implementation of the tensor logic.

## Files

*   **tensor.cpp**: Implements the `Tensor` class methods.
    *   **Constructors**: Initializes storage, shape, and stride.
    *   **Operation Factories**: Methods like `add`, `mul`, `matmul` which:
        1.  Check input shapes/types.
        2.  Dispatch to the appropriate `ops` kernel.
        3.  If gradients are required, create a `Node` in the autograd graph with the appropriate backward closure.
    *   **Autograd**: `backward()` triggers the reverse-mode automatic differentiation from the given tensor (usually loss).
