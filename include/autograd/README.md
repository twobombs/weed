# Autograd

This module provides the infrastructure for automatic differentiation and optimization in Weed. It implements reverse-mode automatic differentiation (backpropagation) through a computation graph, enabling gradient-based training of neural networks.

## Architecture

The autograd system consists of three main components:
1. **Computation Graph Nodes** - Track operations and their gradients
2. **Loss Functions** - Define the objective to minimize
3. **Optimizers** - Update parameters based on gradients

## Files

### [`node.hpp`](node.hpp)
Defines the core computation graph node structure.

#### Struct: `Node`
Represents a single operation in the autograd computation graph.

| Member | Type | Description |
|--------|------|-------------|
| `parents` | `std::vector<TensorPtr>` | Input tensors to this operation |
| `backward` | `std::function<void()>` | Gradient computation function |

| Constructor | Description |
|-------------|-------------|
| `Node(parents, backward)` | Creates a node with parent tensors and backward function |

**Key Behavior:**
- Constructor automatically calls `make_gradient()` on all parent tensors
- The `backward` function computes gradients using the chain rule
- Nodes are created implicitly when operations are performed on tensors with `requires_grad=true`

**Usage Example:**


### [`adam.hpp`](adam.hpp)
Implements the Adam (Adaptive Moment Estimation) optimizer.

#### Struct: `Adam`
Holds optimizer state and hyperparameters.

| Member | Type | Default | Description |
|--------|------|---------|-------------|
| `lr` | `real1` | 0.001 | Learning rate |
| `beta1` | `real1` | 0.9 | First moment decay |
| `beta2` | `real1` | 0.999 | Second moment decay |
| `eps` | `real1` | 1e-8 | Numerical stability |

#### Function: `adam_step(model, lr)`
Performs a single Adam optimization step.

**Algorithm:**


**Usage:**


### [`sgd.hpp`](sgd.hpp)
Implements Stochastic Gradient Descent (SGD) optimizer.

#### Function: `sgd_step(model, lr, weight_decay=0)`
Performs a single SGD optimization step.

**Algorithm:**


**Usage:**


### [`mse_loss.hpp`](mse_loss.hpp)
Implements Mean Squared Error (MSE) loss function.

#### Function: `mse_loss(pred, target)`
Computes MSE: `mean((pred - target)^2)`

**Usage:**


### [`bci_loss.hpp`](bci_loss.hpp)
Implements Binary Cross-Entropy (BCE) loss function.

#### Function: `bci_loss(pred, target)`
Computes BCE: `-mean(target * log(pred) + (1 - target) * log(1 - pred))`

**Usage:**


### [`bci_with_logits_loss.hpp`](bci_with_logits_loss.hpp)
Implements Binary Cross-Entropy with Logits (BCEWithLogits) loss.

**Advantages:**
- More numerically stable than applying sigmoid + BCE separately
- Combines sigmoid and BCE into a single operation

#### Function: `bci_with_logits_loss(pred, target)`

### [`cross_entropy_loss.hpp`](cross_entropy_loss.hpp)
Implements Cross-Entropy loss for multi-class classification.

#### Function: `cross_entropy_loss(pred, target)`
Computes cross-entropy: `-sum(target * log(pred))`

**Usage:**


### [`sgd.hpp`](sgd.hpp)
Implements Stochastic Gradient Descent (SGD) optimizer.

#### Function: `sgd_step(model, lr, weight_decay=0)`
Performs a single SGD optimization step.

**Algorithm:**


**Usage:**


### [`zero_grad.hpp`](zero_grad.hpp)
Provides utility for resetting gradients before training steps.

#### Function: `zero_grad(model)`
Sets all parameter gradients to zero.

**Why needed:**
- Gradients accumulate by default in autograd systems
- Must be cleared before each training step

**Usage:**


## Training Loop Pattern



## Gradient Flow



## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).
