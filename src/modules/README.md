# Module Implementations

This directory contains the implementations of the neural network modules.

## Files

*   **linear.cpp**: Implements the `Linear` module.
    *   **Constructor**: Initializes weights and biases. Can perform random initialization (uniform distribution scaled by $1/\sqrt{in\_features}$) or zero initialization.
    *   **forward**: Computes the linear transformation using matrix multiplication (`operator>>`) and addition (for bias).
    *   **parameters**: Returns the weight and bias (if present) parameters for optimization.
*   **dropout.cpp**: Implements the `Dropout` module (randomly zeroing elements).
*   **embedding.cpp**: Implements the `Embedding` module (lookup table).
*   **gru.cpp**: Implements the `GRU` module (Gated Recurrent Unit).
*   **layernorm.cpp**: Implements the `LayerNorm` module.
*   **lstm.cpp**: Implements the `LSTM` module (Long Short-Term Memory).
*   **module.cpp**: Implements base `Module` functionality, including serialization.
*   **sequential.cpp**: Implements the `Sequential` module (container).
