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
*   **dropout.hpp**: Defines the `Dropout` module.
    *   `Dropout`: Randomly zeroes some of the elements of the input tensor with probability `p` during training.
*   **embedding.hpp**: Defines the `Embedding` module.
    *   `Embedding`: A simple lookup table that stores embeddings of a fixed dictionary and size. Uses `SymbolTensor` for input indices.
*   **gru.hpp**: Defines the `GRU` module.
    *   `GRU`: Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
*   **layernorm.hpp**: Defines the `LayerNorm` module.
    *   `LayerNorm`: Applies Layer Normalization over a mini-batch of inputs.
*   **lstm.hpp**: Defines the `LSTM` module.
    *   `LSTM`: Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
*   **relu.hpp**: Defines the `ReLU` module.
    *   `ReLU`: Applies the rectified linear unit function element-wise.
*   **sequential.hpp**: Defines the `Sequential` module.
    *   `Sequential`: A sequential container. Modules will be added to it in the order they are passed in the constructor.
*   **sigmoid.hpp**: Defines the `Sigmoid` module.
    *   `Sigmoid`: Applies the sigmoid function element-wise.
*   **tanh.hpp**: Defines the `Tanh` module.
    *   `Tanh`: Applies the Hyperbolic Tangent (Tanh) function element-wise.
