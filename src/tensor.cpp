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

#include "tensor.hpp"
#include "node.hpp"

#include "add.hpp"
#include "mul.hpp"

#include "cpu_complex_storage.hpp"
#include "cpu_real_storage.hpp"
#include "gpu_complex_storage.hpp"
#include "gpu_real_storage.hpp"

namespace Weed {
Tensor Tensor::allocate_like(const Tensor &orig) {
  Tensor out;
  vecCapIntGpu size = 0U;
  for (size_t i = 0U; i < shape.size(); ++i) {
    size += (orig.shape[i] - 1U) * orig.stride[i];
  }
  switch (orig.storage->dtype) {
  case DType::COMPLEX:
    out.storage = std::make_shared<CpuComplexStorage>(size);
  case DType::REAL:
  default:
    out.storage = std::make_shared<CpuRealStorage>(size);
  }

  return out;
}

Tensor::Tensor(std::vector<vecCapIntGpu> shp, std::vector<vecCapIntGpu> strd,
               bool rg, DType dtype, DeviceTag dtag, int64_t did)
    : grad(nullptr), shape(shp), stride(strd), offset(0U), requires_grad(rg),
      grad_node(nullptr) {
  if (shape.size() != stride.size()) {
    throw std::invalid_argument(
        "Tensor shape vector must have same length as stride vector!");
  }

  vecCapIntGpu size = 0U;
  for (size_t i = 0U; i < shape.size(); ++i) {
    size += (shape[i] - 1U) * stride[i];
  }

  switch (dtype) {
  case DType::COMPLEX:
    switch (dtag) {
    case DeviceTag::GPU:
      storage = std::make_shared<GpuComplexStorage>(size, did);
    case DeviceTag::CPU:
    default:
      storage = std::make_shared<CpuComplexStorage>(size);
    }
  case DType::REAL:
    switch (dtag) {
    case DeviceTag::GPU:
      storage = std::make_shared<GpuRealStorage>(size, did);
    case DeviceTag::CPU:
    default:
      storage = std::make_shared<CpuRealStorage>(size);
    }
  }
}

std::vector<TensorPtr> filterParents(std::vector<TensorPtr> parents) {
  std::vector<TensorPtr> filtered;
  for (TensorPtr p : parents) {
    if (p->requires_grad) {
      filtered.push_back(p);
    }
  }

  return filtered;
}

Tensor Tensor::add(Tensor &a, Tensor &b) {
  Tensor out = allocate_like(a);

  Weed::add(a, b, out);

  if (!a.requires_grad && !b.requires_grad) {
    return out;
  }

  out.requires_grad = true;

  out.grad_node =
      std::make_shared<Node>(filterParents({a.get_ptr(), b.get_ptr()}),
                             [out](std::vector<TensorPtr> parents) {
                               for (TensorPtr in : parents) {
                                 add_inplace(in->grad, out.grad);
                               }
                             });

  return out;
}

Tensor Tensor::mul(Tensor &a, Tensor &b) {
  Tensor out = allocate_like(a);

  Weed::mul(a, b, out);

  if (!a.requires_grad && !b.requires_grad) {
    return out;
  }

  out.requires_grad = true;

  out.grad_node =
      std::make_shared<Node>(filterParents({a.get_ptr(), b.get_ptr()}),
                             [out](std::vector<TensorPtr> parents) {
                               for (TensorPtr in : parents) {
                                 mul_inplace(in->grad, out.grad);
                               }
                             });

  return out;
}
} // namespace Weed
