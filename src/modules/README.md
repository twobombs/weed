# Module Implementations

This directory contains the implementations of the neural network modules.

## Files

*   **linear.cpp**: Implements the `Linear` module.
    *   **Constructor**: Initializes weights and biases. Can perform random initialization (uniform distribution scaled by $1/\sqrt{in\_features}$) or zero initialization.
    *   **forward**: Computes the linear transformation using matrix multiplication (`operator>>`) and addition (for bias).
    *   **parameters**: Returns the weight and bias (if present) parameters for optimization.
