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
struct ClampKernel {
  void cpu(const Tensor &, const real1 &, const real1 &, Tensor &);
  void cpu_grad_real(Tensor &, const Tensor &, const Tensor &, const real1 &,
                     const real1 &);
  void cpu_grad_complex(Tensor &, const Tensor &, const Tensor &, const real1 &,
                        const real1 &);
  void cpu_grad_mixed(Tensor &, const Tensor &, const Tensor &, const real1 &,
                      const real1 &);
#if ENABLE_GPU
  void gpu(const Tensor &, const real1 &, const real1 &, Tensor &);
  void gpu_grad_real(Tensor &, const Tensor &, const Tensor &, const real1 &,
                     const real1 &);
  void gpu_grad_complex(Tensor &, const Tensor &, const Tensor &, const real1 &,
                        const real1 &);
  void gpu_grad_mixed(Tensor &, const Tensor &, const Tensor &, const real1 &,
                      const real1 &);
#endif
  void clamp(const Tensor &, const real1 &, const real1 &, Tensor &);
  void clamp_grad(Tensor &, const Tensor &, const Tensor &, const real1 &,
                  const real1 &);
};

extern ClampKernel clamp_kernel;

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
