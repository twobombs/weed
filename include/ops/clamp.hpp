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
 * Element-wise clamp
 */
void clamp(const Tensor &a, const real1 &l, const real1 &h, Tensor &out);
/**
 * Element-wise clamp gradient
 */
void clamp_grad(Tensor &din, const Tensor &in, const Tensor &dout,
                const real1 &l, const real1 &h);
} // namespace Weed
