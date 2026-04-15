# Weed Module Implementations

This directory contains the implementation files for neural network modules in the Weed library. These files implement the forward pass, parameter management, and serialization for each layer type.

## Overview

The `modules/` directory implements neural network layers including:
- Basic layers (Linear, Sequential, Dropout)
- RNN variants (GRU, LSTM)
- Attention mechanisms (Multi-head attention)
- Transformer components
- Normalization layers
- Quantum modules

## Files

### [`module.cpp`](module.cpp) - Base Module Implementation

Implementation of the base `Module` class:

**Class: `Module`**
Base class for all neural network modules:

**Implementation Details:**


### [`linear.cpp`](linear.cpp) - Linear Layer

Implementation of the `Linear` (fully connected) layer:

**Class: `Linear`**


### [`sequential.cpp`](sequential.cpp) - Sequential Container

Implementation of the `Sequential` container:

**Class: `Sequential`**


### [`dropout.cpp`](dropout.cpp) - Dropout

Implementation of Dropout regularization:

**Class: `Dropout`**


### [`layernorm.cpp`](layernorm.cpp) - Layer Normalization

Implementation of Layer Normalization:

**Class: `LayerNorm`**


### `RMSNorm` - RMS Normalization

Implementation of RMS (Root Mean Square) Normalization:

**Class: `RMSNorm`**


### [`embedding.cpp`](embedding.cpp) - Embedding Layer

Implementation of the Embedding layer:

**Class: `Embedding`**


### [`gru.cpp`](gru.cpp) - GRU

Implementation of the GRU (Gated Recurrent Unit):

**Class: `GRU`**


### [`lstm.cpp`](lstm.cpp) - LSTM

Implementation of the LSTM (Long Short-Term Memory):

**Class: `LSTM`**


### [`positional_encoding.cpp`](positional_encoding.cpp) - Positional Encoding

Implementation of sinusoidal positional encoding:

**Class: `PositionalEncoding`**


### [`multihead_attention.cpp`](multihead_attention.cpp) - Multi-Head Attention

Implementation of Multi-Head Self-Attention:

**Class: `MultiHeadAttention`**


### [`transformer_encoder_layer.cpp`](transformer_encoder_layer.cpp) - Transformer Encoder

Implementation of Transformer Encoder Layer:

**Class: `TransformerEncoderLayer`**


### [`qrack_neuron.cpp`](qrack_neuron.cpp) - Quantum Neuron

Implementation of Quantum Neuron:

**Class: `QrackNeuron`**


### [`qrack_neuron_layer.cpp`](qrack_neuron_layer.cpp) - Quantum Neuron Layer

Implementation of Quantum Neuron Layer:

**Class: `QrackNeuronLayer`**


### [`migrate_cpu.cpp`](migrate_cpu.cpp) - CPU Migration

Implementation of CPU migration module:

**Class: `MigrateCPU`**


### [`migrate_gpu.cpp`](migrate_gpu.cpp) - GPU Migration

Implementation of GPU migration module:

**Class: `MigrateGPU`**


## Serialization Format

### Module Serialization



### Example: Linear Layer



## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

## Additional Files

### [`qwen_tokenizer.cpp`](qwen_tokenizer.cpp)
Qwen byte-level BPE tokenizer.

### [`qwen_decoder_layer.cpp`](qwen_decoder_layer.cpp)
Qwen-style decoder layer implementation.

### [`qwen.cpp`](qwen.cpp)
Qwen model wrapper.

### [`qwen_model.cpp`](qwen_model.cpp)
Qwen complete model architecture.

### [`rope.cpp`](rope.cpp)
Rotary Positional Embedding (RoPE).

### [`swiglu.cpp`](swiglu.cpp)
SwiGLU activation function.

### [`learned_positional_encoding.cpp`](learned_positional_encoding.cpp)
Learnable Positional Encoding.
