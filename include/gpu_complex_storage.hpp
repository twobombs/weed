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

#include "gpu_device.hpp"
#include "storage.hpp"

#if !ENABLE_OPENCL && !ENABLE_CUDA
#error GPU files were included without either OpenCL and CUDA enabled.
#endif

#include <list>

namespace Weed {
struct GpuComplexStorage : Storage {
  BufferPtr buffer;
  GpuDevicePtr gpu;

  GpuComplexStorage(vecCapIntGpu n, int64_t did) : buffer(nullptr) {
    device = DeviceTag::GPU;
    dtype = DType::COMPLEX;
    size = n;
  }

  ~GpuComplexStorage() {}
};
} // namespace Weed
