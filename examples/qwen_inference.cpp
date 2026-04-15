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

/**
 * Qwen Model Inference Example
 * 
 * This example demonstrates how to:
 * 1. Load a Qwen model from serialized format
 * 2. Use the tokenizer to encode/decode text
 * 3. Run inference on the model
 * 4. Generate text autoregressively
 */

#include "modules/qwen_model.hpp"
#include "modules/qwen_tokenizer.hpp"
#include "storage/typed_storage.hpp"
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

using namespace Weed;

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " <model_file> <tokenizer_vocab> [input_text] [--cpu|--gpu]\n";
    std::cout << "\nArguments:\n";
    std::cout << "  model_file        - Path to the serialized Weed model file\n";
    std::cout << "  tokenizer_vocab   - Path to the tokenizer vocabulary file\n";
    std::cout << "  input_text        - Optional input text for inference\n";
    std::cout << "  --cpu             - Force CPU mode (default)\n";
    std::cout << "  --gpu             - Force GPU mode\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << program << " model.weed vocab.json \"Hello, world!\"\n";
    std::cout << "  " << program << " model.weed vocab.json --gpu\n";
}

int main(int argc, char* argv[]) {
    std::string model_file;
    std::string vocab_file;
    std::string input_text = "The quick brown fox jumps over the lazy dog";
    
    // Parse command-line arguments (input text if provided)
    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg[0] != '-') {
            // Assume it's the input text if not a flag
            input_text = arg;
        }
    }
    
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    model_file = argv[1];
    vocab_file = argv[2];
    
    std::cout << "=== Qwen Model Inference Example ===\n\n";
    
    // Load tokenizer
    std::cout << "Loading tokenizer from: " << vocab_file << "\n";
    QwenTokenizer tokenizer(vocab_file);
    std::cout << "Vocabulary size: " << tokenizer.get_vocab_size() << "\n\n";
    
    // Encode input text
    std::cout << "Input text: " << input_text << "\n";
    auto token_ids = tokenizer.encode(input_text);
    std::cout << "Token IDs: ";
    for (size_t i = 0; i < std::min(token_ids.size(), size_t(20)); ++i) {
        std::cout << token_ids[i] << " ";
    }
    if (token_ids.size() > 20) {
        std::cout << "... (" << token_ids.size() << " total tokens)";
    }
    std::cout << "\n\n";
    
    // Load model
    std::cout << "Loading model from: " << model_file << "\n";
    std::ifstream model_stream(model_file, std::ios::binary);
    if (!model_stream.is_open()) {
        std::cerr << "Error: Could not open model file: " << model_file << "\n";
        return 1;
    }
    
    try {
        auto model = QwenModel::load(model_stream);
        model_stream.close();
        
        // Force CPU mode immediately after loading to avoid GPU device detection
        // This ensures the model works on CPU-only systems
        std::cout << "Using CPU mode (default)\n";
        model->migrate_cpu();
        
        std::cout << "Model loaded successfully!\n";
        std::cout << "  Vocabulary size: " << model->vocab_size << "\n";
        std::cout << "  Hidden size: " << model->hidden_size << "\n";
        std::cout << "  Number of layers: " << model->num_layers << "\n";
        std::cout << "  Number of heads: " << model->num_heads << "\n";
        std::cout << "  Number of KV heads: " << model->num_kv_heads << "\n";
        std::cout << "  Intermediate size: " << model->intermediate_size << "\n";
        std::cout << "  Head dimension: " << model->head_dim << "\n";
        std::cout << "  Max sequence length: " << model->max_seq_len << "\n\n";
        
        // Convert input to tensor
        auto input_tensor = tokenizer.text_to_tensor(input_text);
        
        // Run inference
        std::cout << "Running inference...\n";
        auto start = std::chrono::high_resolution_clock::now();
        
        auto logits = model->forward(input_tensor);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "Inference completed in " << duration.count() << " microseconds\n";
        std::cout << "Output shape: (";
        for (size_t i = 0; i < logits->shape.size(); ++i) {
            std::cout << logits->shape[i];
            if (i < logits->shape.size() - 1) std::cout << ", ";
        }
        std::cout << ")\n\n";
        
        // Get top-5 tokens for first position
        std::cout << "Top-5 predicted tokens for last position:\n";
        tcapint seq_len = logits->shape[logits->shape.size() - 2];
        tcapint vocab_size = logits->shape[logits->shape.size() - 1];
        
        // Get logits for last token
        auto storage_ptr = static_cast<TypedStorage<real1>*>(logits->storage.get());
        tcapint offset = (seq_len - 1) * vocab_size;
        
        // Create pairs of (logit, index) for sorting
        std::vector<std::pair<real1, tcapint>> logit_pairs(vocab_size);
        for (tcapint i = 0; i < vocab_size; ++i) {
            logit_pairs[i].first = (*storage_ptr)[offset + i];
            logit_pairs[i].second = i;
        }
        
        // Sort by logit value (descending)
        std::sort(logit_pairs.begin(), logit_pairs.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Print top 5
        for (int i = 0; i < std::min<int>(5, vocab_size); ++i) {
            std::cout << "  Token " << logit_pairs[i].second 
                      << " (logit: " << logit_pairs[i].first << ")\n";
        }
        
        // Generate some text
        std::cout << "\nGenerating text (10 tokens)...\n";
        auto generated = model->generate(token_ids, 10, 1.0f);
        
        std::cout << "Generated token IDs: ";
        for (size_t i = token_ids.size(); i < generated.size(); ++i) {
            std::cout << generated[i] << " ";
        }
        std::cout << "\n\n";
        
        // Decode generated tokens
        std::string decoded = tokenizer.decode(generated);
        std::cout << "Decoded text: " << decoded << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading or running model: " << e.what() << "\n";
        return 1;
    }
    
    std::cout << "\n=== Example Complete ===\n";
    return 0;
}
