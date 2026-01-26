//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.
//
// Weed is for minimalist AI/ML inference and backprogation in the style of
// Qrack.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#include "common/parallel_for.hpp"

#if defined(_WIN32) && !defined(__CYGWIN__)
#include <direct.h>
#endif

#if ENABLE_PTHREAD
#include <atomic>
#include <future>

#define DECLARE_ATOMIC_vecCapInt() std::atomic<vecCapIntGpu> idx;
#define ATOMIC_ASYNC(...)                                                      \
    std::async(std::launch::async, [__VA_ARGS__]()
#define ATOMIC_INC() i = idx++;
#endif

namespace Weed {

ParallelFor::ParallelFor()
#if ENABLE_ENV_VARS
    : pStride(getenv("WEED_PSTRIDEPOW")
                  ? pow2Gpu((vecLenInt)std::stoi(
                        std::string(getenv("WEED_PSTRIDEPOW"))))
                  : pow2Gpu((vecLenInt)PSTRIDEPOW))
#else
    : pStride(pow2Gpu((vecLenInt)PSTRIDEPOW))
#endif
#if ENABLE_PTHREAD
      ,
      numCores(std::thread::hardware_concurrency())
#else
      ,
      numCores(1U)
#endif
{
  const vecLenInt pStridePow = log2Gpu(pStride);
  const vecLenInt minStridePow =
      (numCores > 1U) ? (vecLenInt)pow2Gpu(log2Gpu(numCores - 1U)) : 0U;
  dispatchThreshold =
      (pStridePow > minStridePow) ? (pStridePow - minStridePow) : 0U;
}

void ParallelFor::par_for(const vecCapIntGpu begin, const vecCapIntGpu end,
                          ParallelFunc fn) {
  par_for_inc(
      begin, end - begin, [](const vecCapIntGpu &i) { return i; }, fn);
}

#if ENABLE_PTHREAD
/*
 * Iterate through the permutations a maximum of end-begin times, allowing the
 * caller to control the incrementation offset through 'inc'.
 */
void ParallelFor::par_for_inc(const vecCapIntGpu begin,
                              const vecCapIntGpu itemCount, IncrementFunc inc,
                              ParallelFunc fn) {
  const vecCapIntGpu Stride = pStride;
  unsigned threads = (unsigned)(itemCount / pStride);
  if (threads > numCores) {
    threads = numCores;
  }

  if (threads <= 1U) {
    const vecCapIntGpu maxLcv = begin + itemCount;
    for (vecCapIntGpu j = begin; j < maxLcv; ++j) {
      fn(inc(j), 0U);
    }

    return;
  }

  DECLARE_ATOMIC_vecCapInt();
  idx = 0U;
  std::vector<std::future<void>> futures;
  futures.reserve(threads);
  for (unsigned cpu = 0U; cpu != threads; ++cpu) {
        futures.emplace_back(ATOMIC_ASYNC(cpu, &idx, &begin, &itemCount, &Stride, inc, fn) {
      for (;;) {
        vecCapIntGpu i;
        ATOMIC_INC();
        const vecCapIntGpu l = i * Stride;
        if (l >= itemCount) {
          break;
        }
        const vecCapIntGpu maxJ =
            ((l + Stride) < itemCount) ? Stride : (itemCount - l);
        for (vecCapIntGpu j = 0U; j < maxJ; ++j) {
          fn(inc(begin + j + l), cpu);
        }
      }
        }));
  }

  for (std::future<void> &future : futures) {
    future.get();
  }
}
#else
/*
 * Iterate through the permutations a maximum of end-begin times, allowing the
 * caller to control the incrementation offset through 'inc'.
 */
void ParallelFor::par_for_inc(const vecCapIntGpu begin,
                              const vecCapIntGpu itemCount, IncrementFunc inc,
                              ParallelFunc fn) {
  const vecCapIntGpu maxLcv = begin + itemCount;
  for (vecCapIntGpu j = begin; j < maxLcv; ++j) {
    fn(inc(j), 0U);
  }
}
#endif

ParallelFor pfControl;
} // namespace Weed
