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

#ifndef QWEN_TOKENIZER_HPP
#define QWEN_TOKENIZER_HPP

#include <string>
#include <vector>
#include <map>
#include <memory>
#include "tensors/tensor.hpp"

namespace Weed {

/**
 * @brief QwenTokenizer - Tokenizer for Qwen language models
 * 
 * This class provides tokenization functionality for Qwen models,
 * including vocabulary loading, encoding, and decoding.
 */
class QwenTokenizer {
public:
    /**
     * @brief Construct a new Qwen Tokenizer with default settings
     */
    QwenTokenizer();

    /**
     * @brief Construct a new Qwen Tokenizer with vocabulary file
     * 
     * @param vocab_file Path to vocabulary file (JSON format)
     * @param merges_file Path to merges file (BPE merges)
     * @param max_length_ Maximum sequence length
     */
    QwenTokenizer(const std::string& vocab_file,
                  const std::string& merges_file = "",
                  tcapint max_length_ = 2048);

    /**
     * @brief Encode text to token IDs
     * 
     * @param text Input text
     * @return std::vector<tcapint> Token IDs
     */
    std::vector<tcapint> encode(const std::string& text) const;

    /**
     * @brief Encode text with optional special tokens
     * 
     * @param text Input text
     * @param add_special_tokens Whether to add BOS/EOS tokens
     * @return std::vector<tcapint> Token IDs
     */
    std::vector<tcapint> encode(const std::string& text,
                                bool add_special_tokens) const;

    /**
     * @brief Decode token IDs to text
     * 
     * @param token_ids Token IDs
     * @return std::string Decoded text
     */
    std::string decode(const std::vector<tcapint>& token_ids) const;

    /**
     * @brief Decode token IDs with optional special token skipping
     * 
     * @param token_ids Token IDs
     * @param skip_special_tokens Whether to skip special tokens
     * @return std::string Decoded text
     */
    std::string decode(const std::vector<tcapint>& token_ids,
                       bool skip_special_tokens) const;

    /**
     * @brief Get token ID for a token string
     * 
     * @param token Token string
     * @return tcapint Token ID
     */
    tcapint get_token_id(const std::string& token) const;

    /**
     * @brief Get token string for a token ID
     * 
     * @param token_id Token ID
     * @return std::string Token string
     */
    std::string get_token(tcapint token_id) const;

    /**
     * @brief Get vocabulary size
     * @return tcapint Vocabulary size
     */
    tcapint get_vocab_size() const;

    /**
     * @brief Convert text to tensor
     * 
     * @param text Input text
     * @return TensorPtr Tensor of token IDs
     */
    TensorPtr text_to_tensor(const std::string& text) const;

    /**
     * @brief Convert tensor to text
     * 
     * @param tensor Tensor of token IDs
     * @return std::string Decoded text
     */
    std::string tensor_to_text(const TensorPtr tensor) const;

private:
    std::map<std::string, tcapint> vocab;
    std::map<tcapint, std::string> reverse_vocab;
    std::map<std::pair<std::string, std::string>, std::string> merge_table;

    tcapint max_length;
    bool use_bpe;

    tcapint bos_token_id;
    tcapint eos_token_id;
    tcapint pad_token_id;
    tcapint unk_token_id;

    std::string bos_token;
    std::string eos_token;
    std::string pad_token;
    std::string unk_token;

    bool load_vocab(const std::string& vocab_file);
    bool load_merges(const std::string& merges_file);
};

using QwenTokenizerPtr = std::shared_ptr<QwenTokenizer>;

} // namespace Weed

#endif // QWEN_TOKENIZER_HPP
