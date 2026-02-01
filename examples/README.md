# Examples

This directory contains example code demonstrating how to use the Weed library.

## Files

*   **nor.cpp**: A simple example that trains a small neural network to learn the logical NOR function.
    *   It demonstrates how to:
        *   Create `Tensor` objects for input data and labels.
        *   Instantiate `Linear` layers and extract their parameters.
        *   Use the `Adam` optimizer.
        *   Construct a forward pass using activation functions (`relu`, `sigmoid`).
        *   Calculate loss (`mse_loss` or `bci_loss`).
        *   Perform backpropagation (`Tensor::backward`) and optimizer steps (`adam_step`).
        *   Reset gradients (`zero_grad`).
