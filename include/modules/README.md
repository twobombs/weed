# Modules

This directory contains neural network module definitions, providing composable layers for building models.

## Files

*   **module.hpp**: Defines the abstract `Module` base class.
    *   `forward`: Pure virtual function to perform the forward pass of the module.
    *   `parameters`: Pure virtual function to return a list of trainable parameters (`ParameterPtr`) in the module.
*   **linear.hpp**: Defines the `Linear` module (fully connected layer).
    *   `Linear`: Represents a linear transformation $y = xW + b$.
        *   Manages `weight` and optional `bias` parameters.
        *   Supports random initialization or zero initialization.
        *   Supports both real and complex data types.
