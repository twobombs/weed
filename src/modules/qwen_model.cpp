//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of
// Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.
//
//////////////////////////////////////////////////////////////////////////////////////

#include "modules/qwen_model.hpp"
#include "tensors/real_tensor.hpp"
#include "tensors/parameter.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace Weed {

QwenModel::QwenModel(tcapint vocab_size_, tcapint hidden_size_, tcapint num_layers_,
                     tcapint num_heads_, tcapint num_kv_heads_, tcapint intermediate_size_,
                     tcapint max_seq_len_, DType dtype, DeviceTag device, int64_t device_id)
    : Module(QWEN_MODEL_T),
      vocab_size(vocab_size_),
      hidden_size(hidden_size_),
      num_layers(num_layers_),
      num_heads(num_heads_),
      num_kv_heads(num_kv_heads_),
      intermediate_size(intermediate_size_),
      head_dim(hidden_size_ / num_heads_),
      max_seq_len(max_seq_len_),
      token_embedding(std::make_shared<Embedding>(vocab_size_, hidden_size_, dtype, device, device_id)),
      decoder(),
      final_norm(std::make_shared<RMSNorm>(hidden_size_, -1)),
      lm_head(std::make_shared<Linear>(hidden_size_, vocab_size_, false, true, dtype, device, device_id)),
      causal_mask(),
      causal_mask_initialized(false),
      current_seq_len(0) {
    
    _init_decoder();
}

void QwenModel::_init_decoder() {
    std::vector<ModulePtr> layers;
    
    for (tcapint i = 0; i < num_layers; ++i) {
        auto layer = std::make_shared<QwenDecoderLayer>(
            hidden_size,
            num_heads,
            num_kv_heads,
            intermediate_size,
            max_seq_len
        );
        layers.push_back(layer);
    }
    
    decoder = std::make_shared<Sequential>(layers);
}

void QwenModel::_init_causal_mask() const {
    if (!causal_mask_initialized) {
        causal_mask = _create_causal_mask(max_seq_len);
        causal_mask_initialized = true;
    }
}

TensorPtr QwenModel::_create_causal_mask(tcapint seq_len) const {
    // Create upper triangular mask with -inf for masked positions
    std::vector<real1> mask_data(seq_len * seq_len, -1e9f);
    
    for (tcapint i = 0; i < seq_len; ++i) {
        for (tcapint j = 0; j <= i; ++j) {
            mask_data[i * seq_len + j] = 0.0f;  // Keep lower triangle (past positions)
        }
    }
    
    auto mask = Tensor::zeros(
        std::vector<tcapint>{1, 1, seq_len, seq_len},
        false,  // not read-only
        true,   // is scalar
        DType::REAL,
        DeviceTag::CPU,
        -1
    );
    
    // Copy data to storage
    auto storage_ptr = static_cast<TypedStorage<real1>*>(mask->storage.get());
    for (size_t i = 0; i < mask_data.size(); ++i) {
        storage_ptr->write(i, mask_data[i]);
    }
    
    return mask;
}

TensorPtr QwenModel::_apply_causal_mask(const TensorPtr attn_scores, tcapint seq_len) const {
    _init_causal_mask();
    
    // Get the causal mask for current sequence length
    TensorPtr mask = causal_mask;
    
    // For now, return attn_scores as-is (causal masking should be done in attention)
    // The QwenDecoderLayer handles causal masking internally via RoPE
    return attn_scores;
}

void QwenModel::reset_cache() {
    past_key_values_k.clear();
    past_key_values_v.clear();
    current_seq_len = 0;
    
    for (const auto& layer : decoder->layers) {
        layer->reset_cache();
    }
}

void QwenModel::set_max_kv_seq_len(tcapint m) {
    max_seq_len = m;
    causal_mask_initialized = false;
    reset_cache();
    
    for (const auto& layer : decoder->layers) {
        layer->set_max_kv_seq_len(m);
    }
}

TensorPtr QwenModel::forward(const SymbolTensorPtr token_ids) const {
    // Token embedding: (batch, seq_len) -> (batch, seq_len, hidden_size)
    TensorPtr embedded = token_embedding->forward(token_ids);
    
    // Forward through decoder layers
    TensorPtr hidden = decoder->forward(embedded);
    
    // Apply final RMSNorm
    hidden = final_norm->forward(hidden);
    
    // Project to vocabulary logits
    TensorPtr logits = lm_head->forward(hidden);
    
    return logits;
}

TensorPtr QwenModel::forward(const TensorPtr hidden_states) {
    // Apply final RMSNorm
    TensorPtr normalized = final_norm->forward(hidden_states);
    
    // Project to vocabulary logits
    TensorPtr logits = lm_head->forward(normalized);
    
    return logits;
}

tcapint QwenModel::_sample_logits(const TensorPtr logits, float temperature) const {
    // Get logits as CPU tensor
    TensorPtr cpu_logits = logits->cast(DeviceTag::CPU);
    
    // Apply temperature scaling and convert to probabilities
    tcapint vocab_size = cpu_logits->shape[cpu_logits->shape.size() - 1];
    std::vector<real1> scaled_logits;
    
    // Access storage directly using TypedStorage
    auto& typed_storage = static_cast<TypedStorage<real1>&>(*cpu_logits->storage);
    for (tcapint i = 0; i < vocab_size; ++i) {
        real1 logit = typed_storage[i];
        scaled_logits.push_back(logit / temperature);
    }
    
    // Convert to probabilities (softmax)
    std::vector<real1> probs;
    real1 max_logit = *std::max_element(scaled_logits.begin(), scaled_logits.end());
    real1 sum_exp = 0.0f;
    
    for (real1 logit : scaled_logits) {
        real1 exp_val = std::exp(logit - max_logit);
        probs.push_back(exp_val);
        sum_exp += exp_val;
    }
    
    for (real1& p : probs) {
        p /= sum_exp;
    }
    
    // Sample from distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    
    return static_cast<tcapint>(dist(gen));
}

std::vector<tcapint> QwenModel::generate(const std::vector<tcapint>& token_ids,
                                          tcapint max_tokens,
                                          float temperature) const {
    std::vector<tcapint> generated = token_ids;
    
    // Convert to symbol tensor for first forward pass
    tcapint batch_size = 1;
    tcapint initial_seq_len = token_ids.size();
    
    auto symbol_tensor = std::make_shared<SymbolTensor>(
        std::vector<tcapint>{batch_size, initial_seq_len},
        std::vector<tcapint>{initial_seq_len, 1},
        false,  // not read-only
        DeviceTag::CPU,
        -1,
        true    // is scalar
    );
    
    // Forward pass through model
    TensorPtr logits = forward(symbol_tensor);
    
    // Sample next token
    tcapint next_token = _sample_logits(logits, temperature);
    generated.push_back(next_token);
    
    // Generate remaining tokens
    for (tcapint i = 1; i < max_tokens; ++i) {
        // Forward pass with single token
        auto single_token = std::make_shared<SymbolTensor>(
            std::vector<tcapint>{batch_size, 1},
            std::vector<tcapint>{1, 1},
            false,  // not read-only
            DeviceTag::CPU,
            -1,
            true    // is scalar
        );
        
        logits = forward(single_token);
        
        // Sample next token
        next_token = _sample_logits(logits, temperature);
        generated.push_back(next_token);
    }
    
    return generated;
}

std::vector<ParameterPtr> QwenModel::parameters() {
    std::vector<ParameterPtr> params;
    
    // Token embedding parameters
    params.insert(params.end(), token_embedding->parameters().begin(),
                  token_embedding->parameters().end());
    
    // Decoder layer parameters
    for (const auto& layer : decoder->layers) {
        params.insert(params.end(), layer->parameters().begin(),
                      layer->parameters().end());
    }
    
    // Final norm parameters
    params.insert(params.end(), final_norm->parameters().begin(),
                  final_norm->parameters().end());
    
    // LM head parameters
    params.insert(params.end(), lm_head->parameters().begin(),
                  lm_head->parameters().end());
    
    return params;
}

void QwenModel::train() {
    token_embedding->train();
    decoder->train();
    final_norm->train();
    lm_head->train();
}

void QwenModel::eval() {
    token_embedding->eval();
    decoder->eval();
    final_norm->eval();
    lm_head->eval();
}

void QwenModel::migrate_cpu() {
    token_embedding->migrate_cpu();
    decoder->migrate_cpu();
    final_norm->migrate_cpu();
    lm_head->migrate_cpu();
}

void QwenModel::migrate_gpu() {
    token_embedding->migrate_gpu();
    decoder->migrate_gpu();
    final_norm->migrate_gpu();
    lm_head->migrate_gpu();
}

void QwenModel::save(std::ostream &os) const {
    // Write module type
    Module::write_module_type(os, mtype);
    
    // Write model config
    os.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    os.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
    os.write(reinterpret_cast<const char*>(&num_layers), sizeof(num_layers));
    os.write(reinterpret_cast<const char*>(&num_heads), sizeof(num_heads));
    os.write(reinterpret_cast<const char*>(&num_kv_heads), sizeof(num_kv_heads));
    os.write(reinterpret_cast<const char*>(&intermediate_size), sizeof(intermediate_size));
    os.write(reinterpret_cast<const char*>(&head_dim), sizeof(head_dim));
    os.write(reinterpret_cast<const char*>(&max_seq_len), sizeof(max_seq_len));
    
    // Save token embedding
    token_embedding->save(os);
    
    // Save decoder layers
    tcapint num_decoder_layers = decoder->layers.size();
    os.write(reinterpret_cast<const char*>(&num_decoder_layers), sizeof(num_decoder_layers));
    for (const auto& layer : decoder->layers) {
        layer->save(os);
    }
    
    // Save final norm
    final_norm->save(os);
    
    // Save LM head
    lm_head->save(os);
}

QwenModelPtr QwenModel::load(std::istream &is) {
    // Note: Module::load() already read the module type before calling this
    // Read model config directly
    tcapint vocab_size, hidden_size, num_layers, num_heads, num_kv_heads, intermediate_size, head_dim, max_seq_len;
    is.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    is.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
    is.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    is.read(reinterpret_cast<char*>(&num_heads), sizeof(num_heads));
    is.read(reinterpret_cast<char*>(&num_kv_heads), sizeof(num_kv_heads));
    is.read(reinterpret_cast<char*>(&intermediate_size), sizeof(intermediate_size));
    is.read(reinterpret_cast<char*>(&head_dim), sizeof(head_dim));
    is.read(reinterpret_cast<char*>(&max_seq_len), sizeof(max_seq_len));
    
    // Create model instance
    auto model = std::make_shared<QwenModel>(
        vocab_size, hidden_size, num_layers, num_heads, num_kv_heads,
        intermediate_size, max_seq_len
    );
    
    // Load token embedding
    model->token_embedding = std::dynamic_pointer_cast<Embedding>(Module::load(is));
    
    // Load decoder layers
    tcapint num_decoder_layers;
    is.read(reinterpret_cast<char*>(&num_decoder_layers), sizeof(num_decoder_layers));
    std::vector<ModulePtr> layers;
    for (tcapint i = 0; i < num_decoder_layers; ++i) {
        layers.push_back(Module::load(is));
    }
    model->decoder = std::make_shared<Sequential>(layers);
    
    // Load final norm
    model->final_norm = std::dynamic_pointer_cast<RMSNorm>(Module::load(is));
    
    // Load LM head
    model->lm_head = std::dynamic_pointer_cast<Linear>(Module::load(is));
    
    return model;
}

} // namespace Weed
