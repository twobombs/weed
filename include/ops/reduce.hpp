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
 * Sum over a tensor index
 */
void reduce(const tcapint &index, const Tensor &a, Tensor &out);
void reduce_grad(const tcapint &index, Tensor &din, const Tensor &a,
                 const Tensor &dout);
void reduce_broadcast(const std::vector<tcapint> stride, const Tensor &a,
                      Tensor &out);
} // namespace Weed
