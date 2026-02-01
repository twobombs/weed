# Autograd

This module provides the necessary components for automatic differentiation and optimization in Weed.

## Files

*   **node.hpp**: Defines the `Node` struct, which is the building block of the autograd computation graph. Each node stores references to its parent tensors and a closure (`std::function`) to execute the backward pass.
*   **adam.hpp**: Implements the **Adam** optimizer.
    *   `Adam`: Structure holding optimizer state (first and second moments) and hyperparameters (`lr`, `beta1`, `beta2`, `eps`).
    *   `adam_step`: Function to perform a single optimization step.
*   **sgd.hpp**: Implements **Stochastic Gradient Descent (SGD)**.
    *   `sgd_step`: Function to perform a simple gradient descent update.
*   **mse_loss.hpp**: Implements **Mean Squared Error (MSE)** loss function (`mse_loss`).
*   **bci_loss.hpp**: Implements **Binary Cross-Entropy (BCI)** loss function (`bci_loss`).
*   **zero_grad.hpp**: Provides the `zero_grad` helper function to reset gradients of parameters to zero before a new training step.
