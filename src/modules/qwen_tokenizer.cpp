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

#include "modules/qwen_tokenizer.hpp"
#include "tensors/real_tensor.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>
#include <set>

namespace Weed {

QwenTokenizer::QwenTokenizer()
    : max_length(2048), use_bpe(true),
      bos_token_id(1), eos_token_id(2), pad_token_id(0), unk_token_id(0),
      bos_token("<|begin_of_text|>"), eos_token("<|end_of_text|>"),
      pad_token("pad_token"), unk_token("unk_token") {}

QwenTokenizer::QwenTokenizer(const std::string& vocab_file,
                             const std::string& merges_file,
                             tcapint max_length_)
    : QwenTokenizer() {
    max_length = max_length_;
    load_vocab(vocab_file);
    if (!merges_file.empty()) {
        load_merges(merges_file);
    }
}

bool QwenTokenizer::load_vocab(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    tcapint id = 0;
    
    while (std::getline(file, line)) {
        // Simple JSON-like parsing for vocab file
        // Format: "token": id or id: "token"
        std::string token;
        
        // Try to parse as "token": id
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);
            
            // Remove quotes from key
            if (key.front() == '"') key = key.substr(1);
            if (key.back() == '"') key.pop_back();
            
            // Remove whitespace from value and parse as int
            value.erase(std::remove_if(value.begin(), value.end(), ::isspace), value.end());
            
            try {
                token = key;
                vocab[token] = id;
                reverse_vocab[id] = token;
                id++;
            } catch (...) {
                continue;
            }
        } else {
            continue;
        }
    }
    
    return true;
}

bool QwenTokenizer::load_merges(const std::string& merges_file) {
    std::ifstream file(merges_file);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        // Skip comments
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string token1, token2;
        iss >> token1 >> token2;
        
        if (!token1.empty() && !token2.empty()) {
            merge_table[{token1, token2}] = token1 + token2;
        }
    }
    
    return true;
}

std::vector<tcapint> QwenTokenizer::encode(const std::string& text) const {
    return encode(text, true);
}

std::vector<tcapint> QwenTokenizer::encode(const std::string& text,
                                            bool add_special_tokens) const {
    std::vector<tcapint> tokens;
    
    // Add BOS token
    if (add_special_tokens) {
        tokens.push_back(bos_token_id);
    }
    
    // Simple character-level tokenization
    // For a production tokenizer, you would use BPE or WordPiece
    std::vector<std::string> word_tokens;
    std::string current_word;
    
    for (char c : text) {
        if (std::isspace(c)) {
            if (!current_word.empty()) {
                word_tokens.push_back(current_word);
                current_word.clear();
            }
            word_tokens.push_back(std::string(1, c));
        } else {
            current_word += c;
        }
    }
    
    if (!current_word.empty()) {
        word_tokens.push_back(current_word);
    }
    
    // Convert words to token IDs
    for (const auto& word : word_tokens) {
        auto it = vocab.find(word);
        if (it != vocab.end()) {
            tokens.push_back(it->second);
        } else {
            // Handle unknown tokens - split into characters
            for (char c : word) {
                std::string char_str(1, c);
                auto char_it = vocab.find(char_str);
                if (char_it != vocab.end()) {
                    tokens.push_back(char_it->second);
                } else {
                    tokens.push_back(unk_token_id);
                }
            }
        }
    }
    
    // Add EOS token
    if (add_special_tokens) {
        tokens.push_back(eos_token_id);
    }
    
    // Truncate if necessary
    if (static_cast<tcapint>(tokens.size()) > max_length) {
        tokens.resize(max_length);
    }
    
    return tokens;
}

std::string QwenTokenizer::decode(const std::vector<tcapint>& token_ids) const {
    return decode(token_ids, false);
}

std::string QwenTokenizer::decode(const std::vector<tcapint>& token_ids,
                                   bool skip_special_tokens) const {
    std::string result;
    
    for (tcapint token_id : token_ids) {
        auto it = reverse_vocab.find(token_id);
        if (it != reverse_vocab.end()) {
            const std::string& token = it->second;
            
            if (skip_special_tokens) {
                if (token == bos_token || token == eos_token ||
                    token == pad_token || token == unk_token) {
                    continue;
                }
            }
            
            result += token;
        }
    }
    
    return result;
}

tcapint QwenTokenizer::get_token_id(const std::string& token) const {
    auto it = vocab.find(token);
    if (it != vocab.end()) {
        return it->second;
    }
    return unk_token_id;
}

std::string QwenTokenizer::get_token(tcapint token_id) const {
    auto it = reverse_vocab.find(token_id);
    if (it != reverse_vocab.end()) {
        return it->second;
    }
    return unk_token;
}

tcapint QwenTokenizer::get_vocab_size() const {
    return static_cast<tcapint>(vocab.size());
}

TensorPtr QwenTokenizer::text_to_tensor(const std::string& text) const {
    std::vector<tcapint> token_ids = encode(text);
    
    // Create a 2D tensor with shape (1, seq_len)
    std::vector<real1> data(token_ids.size());
    for (size_t i = 0; i < token_ids.size(); ++i) {
        data[i] = static_cast<real1>(token_ids[i]);
    }
    
    // Create tensor with data directly
    auto tensor = std::make_shared<Tensor>(
        std::vector<tcapint>{1, static_cast<tcapint>(token_ids.size())},
        std::vector<tcapint>{static_cast<tcapint>(token_ids.size()), 1},
        false,  // not read-only
        true,   // is scalar
        DType::REAL,
        DeviceTag::CPU,
        -1
    );
    
    // Copy data to storage
    auto storage_ptr = static_cast<TypedStorage<real1>*>(tensor->storage.get());
    for (size_t i = 0; i < token_ids.size(); ++i) {
        storage_ptr->write(i, data[i]);
    }
    
    return tensor;
}

std::string QwenTokenizer::tensor_to_text(const TensorPtr tensor) const {
    // Get token IDs from tensor
    std::vector<tcapint> token_ids;
    
    if (tensor->shape.size() == 2) {
        // Remove batch dimension
        tcapint seq_len = tensor->shape[1];
        auto& storage = static_cast<TypedStorage<real1>&>(*tensor->storage);
        for (tcapint i = 0; i < seq_len; ++i) {
            token_ids.push_back(static_cast<tcapint>(storage[i]));
        }
    } else if (tensor->shape.size() == 1) {
        tcapint seq_len = tensor->shape[0];
        auto& storage = static_cast<TypedStorage<real1>&>(*tensor->storage);
        for (tcapint i = 0; i < seq_len; ++i) {
            token_ids.push_back(static_cast<tcapint>(storage[i]));
        }
    }
    
    return decode(token_ids);
}

} // namespace Weed
