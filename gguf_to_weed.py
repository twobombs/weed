#!/usr/bin/env python3
"""
GGUF to Weed Model Converter

Converts Qwen 3.5 GGUF models to Weed's C++ serialization format.
Supports standard tensor extractions and dynamically handles/splits fused QKV tensors 
commonly found in newer LLama.cpp / Mamba hybrid architectures.

Usage:
    python gguf_to_weed.py --input Qwen3.5-2B-Q4_K_S.gguf --output qwen2b.weed
"""

import struct
import argparse
import os
import sys
import numpy as np
import gguf

# Define ModuleType constants (matching C++ enums)
class ModuleType:
    SEQUENTIAL_T = 0
    LINEAR_T = 1
    GELU_T = 2
    RELU_T = 3
    SIGMOID_T = 4
    TANH_T = 5
    SWIGLU_T = 6
    DROPOUT_T = 7
    EMBEDDING_T = 8
    LAYERNORM_T = 9
    GRU_T = 10
    LSTM_T = 11
    MAX_T = 12
    MEAN_T = 13
    MEAN_CENTER_T = 14
    MIN_T = 15
    MULTIHEAD_ATTENTION_T = 16
    POSITIONAL_ENCODING_T = 17
    LEARNED_POSITIONAL_ENCODING_T = 18
    QWEN_DECODER_LAYER_T = 19
    QWEN_T = 20
    RMS_NORM_T = 21
    RESHAPE_T = 22
    ROPE_T = 23
    TRANSFORMER_ENCODER_LAYER_T = 24
    QWEN_MODEL_T = 25
    NONE_MODULE_TYPE = 255

# ============================================================================
# Weed Constants & Binary Writers
# ============================================================================

WEED_MAGIC = 0x44454557  # 'W' 'E' 'E' 'D' (Little Endian)
WEED_VERSION = 1

def w_bool(f, val: bool):
    f.write(struct.pack('<?', val))

def w_size_t(f, val: int):
    f.write(struct.pack('<Q', int(val)))

def w_int64(f, val: int):
    f.write(struct.pack('<q', int(val)))

def w_symint(f, val: int):
    f.write(struct.pack('<q', int(val)))

def w_tcapint(f, val: int):
    f.write(struct.pack('<Q', int(val)))

def w_real1_f(f, val: float):
    f.write(struct.pack('<f', float(val)))

def w_bool(f, val: bool):
    f.write(struct.pack('<?', bool(val)))

def w_uint32(f, val: int):
    f.write(struct.pack('<I', int(val)))

def w_real1_f(f, val: float):
    f.write(struct.pack('<f', float(val)))

class SlicedTensor:
    """Duck-types gguf.ReaderTensor to allow custom slice injections."""
    def __init__(self, name, shape, data):
        self.name = name
        self.shape = shape
        self.data = data

def write_tensor_module(f, tensor, module_type=8):
    """
    Writes a Weed tensor block:
    1. Module type (default: EMBEDDING_T = 8)
    2. Number of dimensions (tcapint)
    3. Shape array (symint x N)
    4. Flat Data array
    """
    # Write module type
    w_symint(f, module_type)
    
    shape = tensor.shape
    w_tcapint(f, len(shape))
    for dim in shape:
        w_symint(f, dim)
    
    data = tensor.data
    if data is None:
        raise ValueError(f"No data available for tensor {tensor.name}")

    # If data is FP16, cast to FP32 for standard modules.
    # If it is a quantized block (e.g. Q4_K_S / uint8), it writes the raw bytes
    if getattr(data, 'dtype', None) == np.float16:
        data = data.astype(np.float32)
    
    if isinstance(data, np.ndarray):
        f.write(data.tobytes())
    else:
        f.write(data)


def write_embedding_module(f, tensor):
    """
    Writes an Embedding module:
    1. Module type (EMBEDDING_T = 8)
    2. num_embeddings
    3. embedding_dim
    4. Weight tensor data (as Parameter format)
    
    Note: GGUF stores embeddings as [embedding_dim, num_embeddings] but Weed expects
    [num_embeddings, embedding_dim]. We need to transpose.
    Also handles quantized data by converting to float32.
    """
    w_symint(f, ModuleType.EMBEDDING_T)
    shape = tensor.shape
    # GGUF stores embeddings transposed: [embedding_dim, num_embeddings]
    # Weed expects: [num_embeddings, embedding_dim]
    num_embeddings = shape[1]
    embedding_dim = shape[0]
    w_tcapint(f, num_embeddings)  # num_embeddings
    w_tcapint(f, embedding_dim)   # embedding_dim
    
    data = tensor.data
    if data is None:
        raise ValueError(f"No data available for tensor {tensor.name}")
    
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        data = np.frombuffer(data, dtype=np.uint8).reshape(shape)
    
    # Transpose if needed (GGUF stores [embedding_dim, num_embeddings])
    if len(shape) == 2 and shape[0] != num_embeddings:
        data = data.T
    
    # Write Parameter format:
    # 1. device_id (-1 for CPU)
    w_symint(f, -1)
    # 2. offset (0)
    w_tcapint(f, 0)
    # 3. shape and stride
    w_tcapint(f, len(shape))
    for dim in shape:
        w_tcapint(f, dim)  # shape
        w_tcapint(f, 1)    # stride (contiguous)
    
    # 4. Storage format:
    # - storage type (1 = REAL_CPU_DENSE) - written as raw binary (4 bytes)
    f.write(struct.pack('<i', 1))
    # - device_id
    w_symint(f, -1)
    # - size
    total_size = 1
    for dim in shape:
        total_size *= dim
    w_tcapint(f, total_size)
    # - data
    if isinstance(data, np.ndarray):
        f.write(data.astype(np.float32).tobytes())
    else:
        f.write(data)


def write_linear_module(f, tensor, in_features, out_features):
    """
    Writes a Linear module:
    1. Module type (LINEAR_T = 1)
    2. in_features
    3. out_features
    4. Weight tensor data (as Parameter format)
    5. is_bias (false for now)
    6. Bias tensor data (if bias exists)
    """
    w_symint(f, ModuleType.LINEAR_T)
    w_tcapint(f, in_features)
    w_tcapint(f, out_features)
    
    data = tensor.data
    if data is None:
        raise ValueError(f"No data available for tensor {tensor.name}")
    
    if getattr(data, 'dtype', None) == np.float16:
        data = data.astype(np.float32)
    
    # Write Parameter format for weight
    w_symint(f, -1)  # device_id
    w_tcapint(f, 0)  # offset
    w_tcapint(f, 2)  # shape dimension count
    w_tcapint(f, in_features)
    w_tcapint(f, out_features)
    w_tcapint(f, 1)  # stride for in_features
    w_tcapint(f, 1)  # stride for out_features
    
    # Storage format
    f.write(struct.pack('<i', 1))  # storage type (REAL_CPU_DENSE) - raw binary
    w_symint(f, -1)  # device_id
    total_size = in_features * out_features
    w_tcapint(f, total_size)
    if isinstance(data, np.ndarray):
        f.write(data.astype(np.float32).tobytes())
    else:
        f.write(data)
    
    # Write is_bias = false
    w_bool(f, False)


def write_rmsnorm_module(f, tensor, hidden_size):
    """
    Writes an RMSNorm module:
    1. Module type (RMS_NORM_T = 21)
    2. axis (-1 for last dimension)
    3. hidden_size
    4. Weight tensor data (as Parameter format)
    """
    w_symint(f, ModuleType.RMS_NORM_T)
    w_symint(f, -1)  # axis
    w_tcapint(f, hidden_size)
    
    data = tensor.data
    if data is None:
        raise ValueError(f"No data available for tensor {tensor.name}")
    
    if getattr(data, 'dtype', None) == np.float16:
        data = data.astype(np.float32)
    
    # Write Parameter format for weight
    w_symint(f, -1)  # device_id
    w_tcapint(f, 0)  # offset
    w_tcapint(f, 1)  # shape dimension count
    w_tcapint(f, hidden_size)
    w_tcapint(f, 1)  # stride
    
    # Storage format
    f.write(struct.pack('<i', 1))  # storage type (REAL_CPU_DENSE) - raw binary
    w_symint(f, -1)  # device_id
    w_tcapint(f, hidden_size)  # size
    if isinstance(data, np.ndarray):
        f.write(data.astype(np.float32).tobytes())
    else:
        f.write(data)


def write_swiglu_module(f, hidden_size, intermediate_size):
    """
    Writes a SwiGLU module:
    1. Module type (SWIGLU_T = 6)
    2. hidden_size
    3. intermediate_size
    """
    w_symint(f, ModuleType.SWIGLU_T)
    w_tcapint(f, hidden_size)
    w_tcapint(f, intermediate_size)


def write_multihead_attention(f, hidden_size, num_heads, num_kv_heads, head_dim):
    """
    Writes a MultiHeadAttention module:
    1. Module type (MULTIHEAD_ATTENTION_T = 16)
    2. mask_val
    3. d_model
    4. num_heads
    5. num_kv_heads
    6. head_dim
    7. use_kv_cache
    8. kv_quant_bits
    9. max_seq_len
    10. W_q, W_k, W_v, W_o (Linear modules)
    11. rope (optional)
    """
    w_symint(f, ModuleType.MULTIHEAD_ATTENTION_T)
    w_real1_f(f, -1e9)  # mask_val (large negative for causal mask)
    w_tcapint(f, hidden_size)  # d_model
    w_tcapint(f, num_heads)
    w_tcapint(f, num_kv_heads)
    w_tcapint(f, head_dim)
    w_bool(f, False)  # use_kv_cache
    w_tcapint(f, 0)  # kv_quant_bits
    w_tcapint(f, 2048)  # max_seq_len


# ============================================================================
# GGUF Metadata & Tensor Extraction
# ============================================================================

def get_gguf_attr(reader: gguf.GGUFReader, possible_suffixes: list, default=None):
    """Safely searches for metadata keys by checking suffixes."""
    for key, field in reader.fields.items():
        for suffix in possible_suffixes:
            if key.endswith(f".{suffix}") or key == suffix:
                parts = field.parts
                if len(parts) == 0:
                    continue
                    
                val = parts[-1]
                if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                    val = val[0]
                if hasattr(val, 'item'):
                    val = val.item()
                if isinstance(val, bytes):
                    val = val.decode('utf-8', errors='ignore')
                    
                return val
                
    if default is not None:
        return default
        
    keys_sample = list(reader.fields.keys())[:20]
    raise ValueError(f"Could not find required GGUF field with suffixes {possible_suffixes}. Sample keys available: {keys_sample}")


def get_layer_tensor(reader: gguf.GGUFReader, layer_idx: int, possible_matches: list):
    """
    Safely extracts a layer tensor using permissive naming conventions.
    Matches whether '.weight' is appended or not.
    """
    layer_markers = [f"blk.{layer_idx}.", f"layers.{layer_idx}.", f"layer.{layer_idx}."]
    for tensor in reader.tensors:
        if any(marker in tensor.name for marker in layer_markers):
            for match in possible_matches:
                if tensor.name.endswith(match) or tensor.name.endswith(match + ".weight"):
                    return tensor
                    
    sample_names = [t.name for t in reader.tensors if any(m in t.name for m in layer_markers)]
    if not sample_names:
        sample_names = [t.name for t in reader.tensors][:10]
    raise KeyError(f"Could not find tensor for layer {layer_idx} matching any of {possible_matches}. Available tensors in this layer block: {sample_names}")


def get_global_tensor(reader: gguf.GGUFReader, possible_matches: list):
    """Safely extracts global tensors like embeddings or output norm."""
    for tensor in reader.tensors:
        for match in possible_matches:
            if tensor.name.endswith(match) or tensor.name.endswith(match + ".weight"):
                return tensor
                
    sample_names = [t.name for t in reader.tensors][:10]
    raise KeyError(f"Could not find global tensor matching {possible_matches}. Available tensors: {sample_names}")


# ============================================================================
# Qwen Serialization
# ============================================================================

def write_qwen_model(f, reader, config):
    print("Writing Model Configuration...")
    # Write QwenModel module type first
    w_symint(f, ModuleType.QWEN_MODEL_T)
    # Write model config directly (matching QwenModel::save)
    # Note: QwenModel::save writes raw binary, not using Serializer
    f.write(struct.pack('<q', config['vocab_size']))
    f.write(struct.pack('<q', config['hidden_size']))
    f.write(struct.pack('<q', config['num_layers']))
    f.write(struct.pack('<q', config['num_heads']))
    f.write(struct.pack('<q', config['num_kv_heads']))
    f.write(struct.pack('<q', config['intermediate_size']))
    head_dim = config['hidden_size'] // config['num_heads']
    f.write(struct.pack('<q', head_dim))
    f.write(struct.pack('<q', 2048))  # max_seq_len

    print("Writing Token Embeddings...")
    token_embd = get_global_tensor(reader, ["token_embd", "embed_tokens"])
    write_embedding_module(f, token_embd)

    for i in range(config['num_layers']):
        print(f"Writing Layer {i}/{config['num_layers']}...")
        
        # Write QwenDecoderLayer module type
        w_symint(f, ModuleType.QWEN_DECODER_LAYER_T)
        w_tcapint(f, config['hidden_size'])  # d_model
        w_tcapint(f, config['num_heads'])
        w_tcapint(f, config['num_kv_heads'])
        
        # Pre-attention LayerNorm (RMSNorm)
        attn_norm = get_layer_tensor(reader, i, ["attn_norm", "input_layernorm"])
        write_rmsnorm_module(f, attn_norm, config['hidden_size'])

        # Attention Projections (Handles separated OR fused QKV blocks)
        try:
            q_proj = get_layer_tensor(reader, i, ["attn_q", "q_proj"])
            k_proj = get_layer_tensor(reader, i, ["attn_k", "k_proj"])
            v_proj = get_layer_tensor(reader, i, ["attn_v", "v_proj"])
        except KeyError:
            # Fused block detected! We byte-slice it based on expected head ratios
            qkv_proj = get_layer_tensor(reader, i, ["attn_qkv", "qkv_proj"])
            
            head_dim = config['hidden_size'] // config['num_heads']
            n_q = int(config['num_heads'] * head_dim)
            n_k = int(config['num_kv_heads'] * head_dim)
            n_v = int(config['num_kv_heads'] * head_dim)
            
            # GGUF shapes represent [fastest_dim, slowest_dim] -> [in_features, out_features]
            dim_out = len(qkv_proj.shape) - 1
            actual_n = int(qkv_proj.shape[dim_out])
            
            bytes_per_row = len(qkv_proj.data) // actual_n
            
            q_data = qkv_proj.data[: int(n_q * bytes_per_row)]
            k_data = qkv_proj.data[int(n_q * bytes_per_row) : int((n_q + n_k) * bytes_per_row)]
            v_data = qkv_proj.data[int(n_q + n_k) * bytes_per_row :]
            
            shape_q = list(qkv_proj.shape)
            shape_q[dim_out] = n_q
            
            shape_k = list(qkv_proj.shape)
            shape_k[dim_out] = n_k
            
            shape_v = list(qkv_proj.shape)
            shape_v[dim_out] = n_v
            
            q_proj = SlicedTensor(qkv_proj.name + "_q", shape_q, q_data)
            k_proj = SlicedTensor(qkv_proj.name + "_k", shape_k, k_data)
            v_proj = SlicedTensor(qkv_proj.name + "_v", shape_v, v_data)
            
        o_proj = get_layer_tensor(reader, i, ["attn_output", "o_proj", "attn_gate", "ssm_out"])
        
        # Write MultiHeadAttention module
        head_dim = config['hidden_size'] // config['num_heads']
        write_multihead_attention(f, config['hidden_size'], config['num_heads'], config['num_kv_heads'], head_dim)
        write_linear_module(f, q_proj, config['hidden_size'], config['hidden_size'])
        write_linear_module(f, k_proj, config['hidden_size'], config['hidden_size'])
        write_linear_module(f, v_proj, config['hidden_size'], config['hidden_size'])
        write_linear_module(f, o_proj, config['hidden_size'], config['hidden_size'])

        # Post-attention LayerNorm (RMSNorm)
        ffn_norm = get_layer_tensor(reader, i, ["ffn_norm", "post_attention_layernorm", "post_attention_norm"])
        write_rmsnorm_module(f, ffn_norm, config['hidden_size'])

        # FFN Projections (SwiGLU)
        gate_proj = get_layer_tensor(reader, i, ["ffn_gate", "gate_proj", "ffn_gate_inp"])
        up_proj   = get_layer_tensor(reader, i, ["ffn_up", "up_proj"])
        down_proj = get_layer_tensor(reader, i, ["ffn_down", "down_proj", "ffn_down_exp"])
        
        # Write SwiGLU module header
        write_swiglu_module(f, config['hidden_size'], config['intermediate_size'])
        
        # Write SwiGLU weights (gate_proj, up_proj, down_proj)
        write_linear_module(f, gate_proj, config['hidden_size'], config['intermediate_size'])
        write_linear_module(f, up_proj, config['hidden_size'], config['intermediate_size'])
        write_linear_module(f, down_proj, config['intermediate_size'], config['hidden_size'])

    print("Writing Final Output Norm...")
    output_norm = get_global_tensor(reader, ["output_norm"])
    write_rmsnorm_module(f, output_norm, config['hidden_size'])

    print("Writing LM Head...")
    try:
        lm_head = get_global_tensor(reader, ["output", "lm_head"])
    except KeyError:
        print("LM Head not found natively, falling back to tied token_embeddings.")
        lm_head = token_embd
    
    # LM Head is a Linear module
    write_linear_module(f, lm_head, config['hidden_size'], config['vocab_size'])


def serialize(input_file, output_file):
    print(f"Loading GGUF file: {input_file}")
    
    reader = gguf.GGUFReader(input_file)
    
    # Build configuration dynamically using integer casting
    config = {
        'vocab_size': int(get_gguf_attr(reader, ["vocabulary_size"], default=151936)),
        'hidden_size': int(get_gguf_attr(reader, ["embedding_length"])),
        'num_layers': int(get_gguf_attr(reader, ["block_count"])),
        'num_heads': int(get_gguf_attr(reader, ["attention.head_count"])),
        'num_kv_heads': int(get_gguf_attr(reader, ["attention.head_count_kv"], default=0)),
        'intermediate_size': int(get_gguf_attr(reader, ["feed_forward_length"], default=0)),
    }
    
    if config['num_kv_heads'] == 0:
        config['num_kv_heads'] = config['num_heads']

    print(f"Model Config Extracted: {config}")

    # QWEN_MODEL_T = 35 (from module_type.hpp)
    QWEN_MODEL_T = 35

    with open(output_file, 'wb') as f:
        w_uint32(f, WEED_MAGIC)
        w_uint32(f, WEED_VERSION)
        w_uint32(f, QWEN_MODEL_T)  # Write module type
        write_qwen_model(f, reader, config)
    
    print(f"Successfully converted {input_file} to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert Qwen 3.5 GGUF to Weed format')
    parser.add_argument('--input', '-i', required=True, help='Input GGUF file path')
    parser.add_argument('--output', '-o', required=True, help='Output .weed file path')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    serialize(args.input, args.output)

if __name__ == "__main__":
    main()
