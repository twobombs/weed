# Examples

This directory contains example code demonstrating how to use the Weed library.

## Files

*   **xor.cpp**: A simple example that trains a small neural network to learn the logical XOR function.
    *   It demonstrates how to:
        *   Create `Tensor` objects for input data and labels.
        *   Define a `Sequential` model with `Linear` layers, `Tanh`, and `Sigmoid` activations.
        *   Use the `Adam` optimizer.
        *   Calculate loss (`bci_loss`).
        *   Perform backpropagation (`Tensor::backward`) and optimizer steps (`adam_step`).
        *   Reset gradients (`zero_grad`).
        *   Save (`save`) and load (`Module::load`) the trained model.
