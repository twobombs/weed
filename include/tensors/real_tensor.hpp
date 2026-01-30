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

#include "storage/real_storage.hpp"
#include "tensors/tensor.hpp"

namespace Weed {
/**
 * Interface to read Tensor as flat and real-valued
 *
 * No new properties or virtual methods are ever added beyond Weed::Tensor in
 * any sub-classes, so it is always possible (though not semantically "safe") to
 * static_cast a Weed::Tensor* based on its offset property to the scalar
 * element to which the Tensor.offset points, based on Tensor.storage->dtype.
 * (Any addition of data members, virtual methods, or multiple inheritance to
 * these types or sub-classes is a breaking change that violates this "unsafe"
 * documented feature.)
 */
struct RealTensor : public Tensor {
  RealTensor(TensorPtr orig) : Tensor(TensorPtr orig) {
    if (storage->dtype != DType::REAL) {
      throw std::domain_error("RealTensor constructor must copy from a "
                              "real-valued generic Tensor!");
    }
  }

  /**
   * Select element at flattened position
   */
  real1 operator[](const tcapint &idx) const {
    tcapint curr = idx;
    tcapint stor = offset;
    for (size_t i = 0U; (i < shape.size()) && curr; ++i) {
      const tcapint &l = shape[i];
      stor += (curr % l) * stride[i];
      curr /= l;
    }

    if (curr) {
      throw std::invalid_argument("RealTensor index out-of-range!");
    }

    return (*static_cast<RealStorage *>(storage.get()))[stor];
  }
};

typedef std::shared_ptr<RealScalar> RealTensorPtr;
} // namespace Weed
