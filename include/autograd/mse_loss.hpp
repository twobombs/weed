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

#include "tensors/tensor.hpp"

namespace Weed {
/**
 * Mean-square error loss
 */
inline TensorPtr mse_loss(TensorPtr y_pred, TensorPtr y_true) {
  return Tensor::mean((y_pred - y_true) * (y_pred - y_true));
}
} // namespace Weed
