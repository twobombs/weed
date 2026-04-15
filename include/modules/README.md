# Weed Neural Network Modules

This directory contains the neural network module definitions for building deep learning models in Weed.

## Overview

The `modules/` directory implements composable neural network building blocks that follow the Weed module interface. Each module supports:
- Forward propagation via `forward()` method
- Parameter extraction via `parameters()` method
- Training/evaluation mode switching
- Serialization/deserialization
- Automatic differentiation integration

## Base Class

### [`module.hpp`](module.hpp) - Module Interface

The `Module` class is the base interface for all neural network modules:

**Key Members:**
- `mtype`: Module type identifier for serialization

**Key Methods:**
- `forward(const TensorPtr)`: Pure virtual - computes output from input
- `forward(const SymbolTensorPtr)`: Symbol tensor forward (for embedding)
- `forward(Qrack::QInterfacePtr)`: Quantum forward (when QRACK available)
- `parameters()`: Returns trainable parameters
- `train()`: Sets training mode (enables gradients, dropout)
- `eval()`: Sets evaluation mode (disables gradients, dropout)
- `save(std::ostream&)`: Serializes module to stream
- `load(std::istream&)`: Static factory for deserialization

**Quantum Support:**
When `QRACK_AVAILABLE` is defined, modules can implement quantum forward methods that operate on `Qrack::QInterfacePtr` for hybrid quantum-classical models.

## Core Layers

### [`linear.hpp`](linear.hpp) / [`linear.cpp`](../src/modules/linear.cpp) - Fully Connected Layer

Implements a linear transformation: `y = xW^T + b`

**Features:**
- Configurable input/output dimensions
- Trainable weight and bias parameters
- Optional activation function support
- GPU acceleration via OpenCL

**Usage:**


### [`sequential.hpp`](sequential.hpp) / [`sequential.cpp`](../src/modules/sequential.cpp) - Sequential Container

Container module for sequential models:

**Features:**
- Maintains ordered list of sub-modules
- Forward passes through all modules in order
- Aggregates parameters from all sub-modules
- Supports nested sequential containers

**Usage:**


### [`dropout.hpp`](dropout.hpp) / [`dropout.cpp`](../src/modules/dropout.cpp) - Dropout Regularization

Drops random elements during training to prevent overfitting:

**Features:**
- Configurable dropout probability
- Only active in training mode
- Scales activations by (1 - p) during training
- No-op in evaluation mode

**Usage:**


### [`layernorm.hpp`](layernorm.hpp) / [`layernorm.cpp`](../src/modules/layernorm.cpp) - Layer Normalization

Normalizes activations across features:

**Features:**
- Learnable scale (gamma) and shift (beta) parameters
- Stable normalization with epsilon
- Works on any number of dimensions

**Usage:**


### [`embedding.hpp`](embedding.hpp) / [`embedding.cpp`](../src/ops/embedding.cpp) - Embedding Lookup

Maps integer indices to dense vectors:

**Features:**
- `forward(SymbolTensorPtr)`: Takes integer indices
- `forward(TensorPtr)`: Takes dense embeddings
- Trainable embedding matrix
- Supports GPU acceleration

**Usage:**


## RNN Variants

### [`gru.hpp`](gru.hpp) / [`gru.cpp`](../src/modules/gru.cpp) - Gated Recurrent Unit

GRU cell for sequence modeling:

**Features:**
- Reset and update gates
- Hidden state management
- Optional batch-first processing
- Supports hidden state initialization

**Usage:**


### [`lstm.hpp`](lstm.hpp) / [`lstm.cpp`](../src/modules/lstm.cpp) - Long Short-Term Memory

LSTM cell with cell state:

**Features:**
- Input, forget, output gates
- Cell state and hidden state
- Optional peephole connections
- Supports hidden/cell state initialization

**Usage:**


## Attention Mechanisms

### [`multihead_attention.hpp`](multihead_attention.hpp) / [`multihead_attention.cpp`](../src/modules/multihead_attention.cpp) - Multi-Head Self-Attention

Implements multi-head attention with optional masking:

**Features:**
- Configurable number of heads
- Query, Key, Value projections
- Scaled dot-product attention
- Optional attention mask
- Dropout on attention weights

**Usage:**


### [`positional_encoding.hpp`](positional_encoding.hpp) / [`positional_encoding.cpp`](../src/modules/positional_encoding.cpp) - Sinusoidal Positional Encoding

Adds fixed sinusoidal positional encodings:

**Features:**
- Pre-computed sinusoidal positions
- No trainable parameters
- Supports max sequence length configuration

**Usage:**


### [`learned_positional_encoding.hpp`](learned_positional_encoding.hpp) / [`learned_positional_encoding.cpp`](../src/modules/learned_positional_encoding.cpp) - Learnable Positional Encoding

Trainable positional embeddings:

**Features:**
- Learnable position vectors
- Configurable max sequence length
- Zero initialization

**Usage:**


### [`rope.hpp`](rope.hpp) / [`rope.cpp`](../src/modules/rope.cpp) - Rotary Positional Embedding (RoPE)

Rotary embeddings for LLMs:

**Features:**
- Rotary transformation based on position
- No additional parameters
- Compatible with attention mechanisms

**Usage:**


## Transformer Components

### [`transformer_encoder_layer.hpp`](transformer_encoder_layer.hpp) / [`transformer_encoder_layer.cpp`](../src/modules/transformer_encoder_layer.cpp) - Transformer Encoder Layer

Complete encoder layer with MHA and feed-forward:

**Features:**
- Multi-head self-attention
- Feed-forward network with GELU
- Layer normalization (pre-norm)
- Residual connections
- Optional attention mask

**Usage:**


### [`qwen.hpp`](qwen.hpp) / [`qwen.cpp`](../src/modules/qwen.cpp) - Qwen Model
Wrapper for Qwen-style models, combining:
- Embedding layer
- Sequential stack of decoder layers
- RMSNorm
- Final vocabulary projection

```cpp
auto model = std::make_shared<Qwen>(vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size);
TensorPtr output = model->forward(input_symbols);
```

### [`qwen_decoder_layer.hpp`](qwen_decoder_layer.hpp) / [`qwen_decoder_layer.cpp`](../src/modules/qwen_decoder_layer.cpp) - Qwen Decoder Layer

Qwen-style decoder layer with KV cache:

**Features:**
- Causal self-attention with KV cache
- RoPE positional encoding
- SwiGLU feed-forward
- RMS normalization
- Configurable max KV sequence length

**Usage:**


### [`qwen_model.hpp`](qwen_model.hpp) / [`qwen_model.cpp`](../src/modules/qwen_model.cpp) - Qwen Model

Complete Qwen decoder-only transformer model architecture:

**Features:**
- Token Embedding mapping token IDs to dense vectors
- Decoder Layers as a stack of QwenDecoderLayer instances
- Final RMSNorm for normalization before output projection
- LM Head mapping hidden states to vocabulary logits
- Causal masking for autoregressive generation
- KV cache support for efficient inference

**Usage:**


### [`qwen_tokenizer.hpp`](qwen_tokenizer.hpp) / [`qwen_tokenizer.cpp`](../src/modules/qwen_tokenizer.cpp) - Qwen Tokenizer

Simple tokenizer implementing basic byte-level BPE tokenization similar to the Qwen tokenizer:

**Features:**
- Vocabulary loading from JSON files
- Encoding text to token IDs
- Decoding token IDs to text
- Special token handling

**Usage:**


## Normalization

### [`rms_norm.hpp`](rms_norm.hpp) - RMS Normalization

Root mean square layer normalization:

**Features:**
- No mean centering (faster than LayerNorm)
- Learnable scale parameter
- Stable with epsilon

**Usage:**


## Quantum Modules

### [`qrack_neuron.hpp`](qrack_neuron.hpp) / [`qrack_neuron.cpp`](../src/modules/qrack_neuron.cpp) - Quantum Neuron

Single quantum neuron with parameterized gates:

**Features:**
- Parameterized rotation gates
- Quantum-classical interface
- Gradient computation via parameter shift

**Usage:**


### [`qrack_neuron_layer.hpp`](qrack_neuron_layer.hpp) / [`qrack_neuron_layer.cpp`](../src/modules/qrack_neuron_layer.cpp) - Quantum Neuron Layer

Layer of quantum neurons:

**Features:**
- Multiple quantum neurons
- Shared or separate parameters
- Batch processing support

**Usage:**


## Utilities

### [`migrate_cpu.hpp`](migrate_cpu.hpp) / [`migrate_cpu.cpp`](../src/modules/migrate_cpu.cpp) - CPU Migration Module

Moves all parameters to CPU:

**Features:**
- Recursive parameter migration
- No-op if already on CPU

**Usage:**


### [`migrate_gpu.hpp`](migrate_gpu.hpp) / [`migrate_gpu.cpp`](../src/modules/migrate_gpu.cpp) - GPU Migration Module

Moves all parameters to GPU:

**Features:**
- Recursive parameter migration
- Configurable device ID
- No-op if already on GPU

**Usage:**


## Serialization

All modules support serialization via:



Serialization format:
1. Module type (4 bytes)
2. Module-specific parameters
3. Parameter tensors (weights, biases)

## Training Workflow



## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

## Additional Files

### [`mean_center.hpp`](mean_center.hpp)
Mean center module.

### [`sigmoid.hpp`](sigmoid.hpp)
Sigmoid activation function module.

### [`max.hpp`](max.hpp)
Max reduction module.

### [`relu.hpp`](relu.hpp)
ReLU activation function module.

### [`swiglu.hpp`](swiglu.hpp)
SwiGLU activation function module.

### [`gelu.hpp`](gelu.hpp)
GELU activation function module.

### [`flatten.hpp`](flatten.hpp)
Flatten operation module.

### [`stddev.hpp`](stddev.hpp)
Standard deviation reduction module.

### [`min.hpp`](min.hpp)
Min reduction module.

### [`reshape.hpp`](reshape.hpp)
Reshape operation module.

### [`logsoftmax.hpp`](logsoftmax.hpp)
LogSoftmax activation function module.

### [`mean.hpp`](mean.hpp)
Mean reduction module.

### [`variance.hpp`](variance.hpp)
Variance reduction module.

### [`softmax.hpp`](softmax.hpp)
Softmax activation function module.

### [`tanh.hpp`](tanh.hpp)
Tanh activation function module.
