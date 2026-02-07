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

#pragma once

#include "autograd/node.hpp"
#include "modules/module.hpp"
#include "tensors/tensor.hpp"

namespace Weed {
/**
 * Convenience wrapper on migration to GPU
 */
struct MigrateGpu : public Module {
  MigrateGpu() : Module(MIGRATE_GPU) {}
  TensorPtr forward(const TensorPtr x) override {
    TensorPtr out = std::make_shared<Tensor>(*(x.get()));
    out->storage = out->storage->gpu();
    out->make_gradient();
    out->grad_node = std::make_shared<Node>(std::vector<TensorPtr> {x}, [x, out] {
      x->grad->storage = (x->storage->device == DeviceTag::CPU) ? out->grad->storage->cpu() : out->grad->storage->gpu();
    });

    return out;
  }
};
typedef std::shared_ptr<MigrateGpu> MigrateGpuPtr;
} // namespace Weed
