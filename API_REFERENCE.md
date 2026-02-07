# Weed API Reference

Weed is a minimalist C++ AI/ML library designed for inference and backpropagation. It features a tensor library with automatic differentiation (autograd), standard deep learning modules, and optimizers.

## Table of Contents

1.  [Tensor API](#tensor-api)
2.  [Autograd System](#autograd-system)
3.  [Module API](#module-api)
4.  [Optimizers](#optimizers)
5.  [Usage Example](#usage-example)

---

## Tensor API

The `Tensor` class is the core data structure in Weed, representing multi-dimensional arrays with support for automatic differentiation.

**Header:** `#include "tensors/tensor.hpp"`

### Types

*   `TensorPtr`: `std::shared_ptr<Tensor>`
*   `real1`: Precision-dependent real number (float/double).
*   `tcapint`: Tensor capacity integer type.

### Creating Tensors

Tensors are typically created using `std::make_shared<Tensor>`.

```cpp
// Create a Tensor with data
std::vector<real1> data = {1.0, 2.0, 3.0, 4.0};
std::vector<tcapint> shape = {2, 2};
std::vector<tcapint> stride = {2, 1}; // Row-major contiguous
bool requires_grad = true;

TensorPtr t = std::make_shared<Tensor>(
    data,
    shape,
    stride,
    requires_grad,
    DeviceTag::CPU
);
```

#### Factory Methods

*   **Zeros**:
    ```cpp
    TensorPtr z = Tensor::zeros({2, 2}, true); // shape, requires_grad
    ```

*   **Ones Like**:
    ```cpp
    TensorPtr o = Tensor::ones_like({2, 2}, true);
    ```

### Operations

Weed supports standard mathematical operations. These operations record the computation graph if inputs require gradients.

#### Arithmetic
*   `add(a, b)` / `operator+`: Element-wise addition.
*   `sub(a, b)` / `operator-`: Element-wise subtraction.
*   `mul(a, b)` / `operator*`: Element-wise multiplication.
*   `div(a, b)` / `operator/`: Element-wise division.
*   `matmul(a, b)`: Matrix multiplication (2D tensors only).

#### Activation Functions
*   `sigmoid(a)`: Sigmoid activation.
*   `tanh(a)`: Hyperbolic tangent activation.
*   `relu(a)`: Rectified Linear Unit.
*   `clamp(a, min, max)`: Clamp values to range.

#### Reduction
*   `sum(a)`: Sum all elements.
*   `mean(a)`: Mean of all elements.
*   `sum(a, axis)`: Sum along an axis.
*   `mean(a, axis)`: Mean along an axis.

### Autograd Interface

*   `backward(TensorPtr loss)`: Static method. Computes gradients for the entire graph ending at `loss`.
*   `grad`: Member variable storing the gradient (as a `TensorPtr`) after `backward()` is called.

---

## Autograd System

Weed uses a dynamic computational graph (Define-by-Run).

**Header:** `#include "autograd/node.hpp"`

Each `Tensor` maintains a `grad_node` (of type `NodePtr`) which points to its parents in the computation graph and contains the backward function.

When `Tensor::backward(loss)` is called:
1.  It seeds the gradient of `loss` with ones.
2.  It performs a topological sort of the graph starting from `loss`.
3.  It executes the `backward` function of each node in reverse topological order, propagating gradients to `grad` fields of parent tensors.

---

## Module API

Modules organize parameters and forward passes.

**Header:** `#include "modules/module.hpp"`

### Base Class: `Module`

All neural network layers inherit from `Module`.

*   `forward(TensorPtr input)`: Defines the computation.
*   `parameters()`: Returns a list of learnable parameters (`ParameterPtr`).
*   `train()` / `eval()`: Switches modes (e.g., for Dropout).
*   `save(ostream)` / `load(istream)`: Serialization.

### Standard Modules

**Header:** `#include "modules/linear.hpp"`, etc.

*   **Linear**: Fully connected layer.
    ```cpp
    auto layer = std::make_shared<Linear>(input_dim, output_dim);
    ```

*   **Sequential**: Containers for a stack of modules.
    ```cpp
    auto model = std::make_shared<Sequential>(std::vector<ModulePtr>{layer1, layer2});
    ```

*   **Activations**: `Sigmoid`, `Tanh` (as Modules).

---

## Optimizers

Optimizers update parameters based on their gradients.

**Header:** `#include "autograd/adam.hpp"`

### Adam

Adaptive Moment Estimation optimizer.

```cpp
Adam opt(learning_rate);
opt.register_parameters(model.parameters());

// In training loop:
adam_step(opt, model.parameters());
```

**Header:** `#include "autograd/zero_grad.hpp"`

*   `zero_grad(params)`: Clears gradients before the next backward pass.

---

## Usage Example

Here is a complete example of training a simple XOR classifier, derived from `examples/xor.cpp`.

```cpp
#include "tensors/tensor.hpp"
#include "modules/linear.hpp"
#include "modules/sequential.hpp"
#include "modules/sigmoid.hpp"
#include "modules/tanh.hpp"
#include "autograd/adam.hpp"
#include "autograd/bci_loss.hpp" // Binary Cross Entropy Loss
#include "autograd/zero_grad.hpp"
#include <vector>
#include <iostream>

using namespace Weed;

// Helper macros for readability (implementation specific)
#define R(v) real1(v)

int main() {
    // 1. Prepare Data (XOR Problem)
    // Inputs: [0,0], [1,0], [0,1], [1,1]
    TensorPtr x = std::make_shared<Tensor>(
        std::vector<real1>{R(0), R(1), R(0), R(1), R(0), R(0), R(1), R(1)},
        std::vector<tcapint>{4, 2}, // Shape: 4 samples, 2 features
        std::vector<tcapint>{1, 4}, // Stride: Column-major in this specific example check
        false, // No gradient needed for input
        DeviceTag::CPU
    );

    // Labels: 0, 1, 1, 0
    TensorPtr y = std::make_shared<Tensor>(
        std::vector<real1>{R(0), R(1), R(1), R(0)},
        std::vector<tcapint>{4, 1},
        std::vector<tcapint>{1, 0},
        false,
        DeviceTag::CPU
    );

    // 2. Build Model
    // 2 inputs -> 4 hidden -> Tanh -> 1 output -> Sigmoid
    Sequential model({
        std::make_shared<Linear>(2, 4),
        std::make_shared<Tanh>(),
        std::make_shared<Linear>(4, 1),
        std::make_shared<Sigmoid>()
    });

    // 3. Setup Optimizer
    auto params = model.parameters();
    Adam opt(R(0.01)); // Learning rate 0.01
    opt.register_parameters(params);

    // 4. Training Loop
    for (int epoch = 1; epoch <= 200; ++epoch) {
        // Forward pass
        TensorPtr y_pred = model.forward(x);

        // Compute loss
        TensorPtr loss = bci_loss(y_pred, y);

        // Backward pass
        Tensor::backward(loss);

        // Update weights
        adam_step(opt, params);

        // Clear gradients for next step
        zero_grad(params);

        // Log
        if (epoch % 10 == 0) {
             // Access scalar value (cast required to storage type)
             real1 loss_val = static_cast<RealScalar *>(loss->storage.get())->get_item();
             std::cout << "Epoch " << epoch << ", Loss: " << loss_val << std::endl;
        }
    }

    // 5. Inference
    model.eval();
    TensorPtr pred = model.forward(x);
    // Access predictions from storage...
}
```
