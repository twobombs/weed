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

#include "modules/qwen.hpp"
#include "modules/qwen_model.hpp"

namespace Weed {

// Stub class for Qwen (used by qwen.cpp)
class Qwen : public QwenModel {
public:
    Qwen(tcapint vocab_size_, tcapint hidden_size_, tcapint num_layers_,
         tcapint num_heads_, tcapint num_kv_heads_, tcapint intermediate_size_,
         tcapint max_seq_len_, DType dtype = DType::REAL,
         DeviceTag device = DeviceTag::CPU, int64_t device_id = -1)
        : QwenModel(vocab_size_, hidden_size_, num_layers_, num_heads_,
                    num_kv_heads_, intermediate_size_, max_seq_len_,
                    dtype, device, device_id) {}

    void save(std::ostream &os) const override {
        QwenModel::save(os);
    }
};

} // namespace Weed
