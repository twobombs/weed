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
#include "tensors/parameter.hpp"

namespace Weed {
/**
 * Standard interface for sequential models of multiple layers
 */
class Sequential : public Module {
  std::vector<ModulePtr> layers;

public:
  TensorPtr forward(TensorPtr x);
  std::vector<ParameterPtr> parameters();
};
} // namespace Weed
