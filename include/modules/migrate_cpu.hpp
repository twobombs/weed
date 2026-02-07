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

#include "modules/module.hpp"
#include "tensors/tensor.hpp"

namespace Weed {
/**
 * Convenience wrapper on migration to CPU
 */
struct MigrateCpu : public Module {
  MigrateCpu() : Module(MIGRATE_CPU) {}
  TensorPtr forward(const TensorPtr x) override;
};
typedef std::shared_ptr<MigrateCpu> MigrateCpuPtr;
} // namespace Weed
