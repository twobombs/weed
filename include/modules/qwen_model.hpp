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

#ifndef QWEN_MODEL_HPP
#define QWEN_MODEL_HPP

#include "modules/module.hpp"
#include "modules/sequential.hpp"
#include "modules/embedding.hpp"
#include "modules/linear.hpp"
#include "modules/rms_norm.hpp"
#include "modules/qwen_decoder_layer.hpp"
#include "tensors/symbol_tensor.hpp"

namespace Weed {

// Forward declaration
class QwenModel;
using QwenModelPtr = std::shared_ptr<QwenModel>;

/**
 * @brief QwenModel - A Qwen-style decoder-only transformer language model
 * 
 * This class implements a Qwen (Qwen2/Qwen3) decoder-only transformer architecture
 * suitable for text generation and language modeling tasks.
 */
class QwenModel : public Module {
public:
    /**
     * @brief Construct a new Qwen Model
     * 
     * @param vocab_size_ Size of the vocabulary
     * @param hidden_size_ Hidden dimension size
     * @param num_layers_ Number of transformer layers
     * @param num_heads_ Number of attention heads
     * @param num_kv_heads_ Number of KV heads (for GQA)
     * @param intermediate_size_ Intermediate size for feed-forward layers
     * @param max_seq_len_ Maximum sequence length
     * @param dtype_ Data type for weights
     * @param device_ Device for computation (CPU/GPU)
     * @param device_id_ Device ID for GPU
     */
    QwenModel(tcapint vocab_size_, tcapint hidden_size_, tcapint num_layers_,
              tcapint num_heads_, tcapint num_kv_heads_, tcapint intermediate_size_,
              tcapint max_seq_len_, DType dtype = DType::REAL,
              DeviceTag device = DeviceTag::CPU, int64_t device_id = -1);

    /**
     * @brief Forward pass through the model
     *
     * @param token_ids Symbol tensor of token IDs (batch, seq_len)
     * @return TensorPtr Logits tensor (batch, seq_len, vocab_size)
     */
    TensorPtr forward(const SymbolTensorPtr token_ids) const;
    using Module::forward;

    /**
     * @brief Forward pass from hidden states
     *
     * @param hidden_states Input hidden states
     * @return TensorPtr Logits tensor
     */
    TensorPtr forward(const TensorPtr hidden_states) override;

    /**
     * @brief Generate text autoregressively
     * 
     * @param token_ids Initial token IDs
     * @param max_tokens Maximum number of tokens to generate
     * @param temperature Sampling temperature (0.0 = greedy)
     * @return std::vector<tcapint> Generated token IDs
     */
    std::vector<tcapint> generate(const std::vector<tcapint>& token_ids,
                                  tcapint max_tokens,
                                  float temperature = 1.0f) const;

    /**
     * @brief Get model parameters
     * @return std::vector<ParameterPtr> List of all parameters
     */
    std::vector<ParameterPtr> parameters() override;

    /**
     * @brief Set training mode
     */
    void train() override;

    /**
     * @brief Set evaluation mode
     */
    void eval() override;

    /**
     * @brief Move model to CPU
     */
    void migrate_cpu() override;

    /**
     * @brief Move model to GPU
     */
    void migrate_gpu() override;

    /**
     * @brief Save model to stream
     * @param os Output stream
     */
    void save(std::ostream &os) const override;

    /**
     * @brief Load model from stream
     * @param is Input stream
     * @return QwenModelPtr Loaded model
     */
    static QwenModelPtr load(std::istream &is);

    // Friend declaration for Module::load
    friend ModulePtr Module::load(std::istream &is);

    // Model configuration
    tcapint vocab_size;
    tcapint hidden_size;
    tcapint num_layers;
    tcapint num_heads;
    tcapint num_kv_heads;
    tcapint intermediate_size;
    tcapint head_dim;
    tcapint max_seq_len;

    /**
     * @brief Reset KV cache
     */
    void reset_cache();

    /**
     * @brief Set maximum KV sequence length
     * @param m New maximum sequence length
     */
    void set_max_kv_seq_len(tcapint m);

private:
    std::shared_ptr<Embedding> token_embedding;
    std::shared_ptr<Sequential> decoder;
    std::shared_ptr<RMSNorm> final_norm;
    std::shared_ptr<Linear> lm_head;

    mutable TensorPtr causal_mask;
    mutable bool causal_mask_initialized;
    tcapint current_seq_len;

    std::vector<TensorPtr> past_key_values_k;
    std::vector<TensorPtr> past_key_values_v;

    void _init_decoder();
    void _init_causal_mask() const;
    TensorPtr _create_causal_mask(tcapint seq_len) const;
    TensorPtr _apply_causal_mask(const TensorPtr attn_scores, tcapint seq_len) const;
    tcapint _sample_logits(const TensorPtr logits, float temperature) const;
};

} // namespace Weed

#endif // QWEN_MODEL_HPP
