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

#include "weed_types.hpp"

#include <vector>

namespace Weed {
struct Tensor;

typedef std::shared_ptr<Tensor> TensorPtr;

struct Tensor : public std::enable_shared_from_this<Tensor> {
  StoragePtr storage;
  StoragePtr grad;

  std::vector<vecCapIntGpu> shape;
  std::vector<vecCapIntGpu> stride;
  vecCapIntGpu offset;

  bool requires_grad;
  NodePtr grad_node;

  Tensor() : offset(0U), requires_grad(false), grad_node(nullptr) {
    validate();
  }

  TensorPtr get_ptr() { return shared_from_this(); }

  Tensor allocate_like(const Tensor &orig);
  void validate() const;

  Tensor add(Tensor &a, Tensor &b);
  Tensor mul(Tensor &a, Tensor &b);
};
} // namespace Weed
