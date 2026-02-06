//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2023. All rights reserved.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or
// https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

#include "shared_api.hpp"

// "qfactory.hpp" pulls in all headers needed to create any type of
// "Qrack::QInterface."
#include "modules/module.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>

#define META_LOCK_GUARD()                                                      \
  const std::lock_guard<std::mutex> meta_lock(meta_operation_mutex)

#define MODULE_LOCK_GUARD(mid)                                                 \
  std::unique_ptr<const std::lock_guard<std::mutex>> module_lock;              \
  if (true) {                                                                  \
    std::lock(meta_operation_mutex, module_results[mid]->mtx);                 \
    const std::lock_guard<std::mutex> metaLock(meta_operation_mutex,           \
                                               std::adopt_lock);               \
    module_lock = std::make_unique<const std::lock_guard<std::mutex>>(         \
        module_results[mid]->mtx, std::adopt_lock);                            \
  }

using namespace Weed;

struct ModuleResult {
  std::mutex mtx;
  ModulePtr m;
  TensorPtr t;
  int error;
  ModuleResult(ModulePtr a) : m(a), t(nullptr), error(0) {}
};
typedef std::unique_ptr<ModuleResult> ModuleResultPtr;

std::mutex meta_operation_mutex;
int meta_error = 0;

std::vector<ModuleResultPtr> module_results;

void _darray_to_creal1_array(double *params, tcapint componentCount,
                             complex *amps) {
  for (tcapint j = 0U; j < componentCount; ++j) {
    amps[j] = complex(real1(params[2U * j]), real1(params[2U * j + 1U]));
  }
}

extern "C" {
MICROSOFT_QUANTUM_DECL int get_error(_In_ const uintw mid) {
  if (meta_error) {
    meta_error = 0;
    return 2;
  }

  return module_results[mid]->error;
}

MICROSOFT_QUANTUM_DECL uintw load_module(_In_ const char *f) {
  META_LOCK_GUARD();

  bool is_success = true;
  ModulePtr m;
  try {
    std::ifstream i(f);
    m = Module::load(i);
    i.close();
    m->eval();
  } catch (const std::exception &ex) {
    std::cout << ex.what() << std::endl;
    is_success = false;
    meta_error = 1;
  }

  uintw id = 0U;
  if (is_success) {
    while ((id < module_results.size()) && module_results[id]) {
      ++id;
    }
    if (id == module_results.size()) {
      module_results.push_back(
          std::unique_ptr<ModuleResult>(new ModuleResult(m)));
    } else {
      module_results[id] = std::unique_ptr<ModuleResult>(new ModuleResult(m));
    }
  }

  return id;
}

MICROSOFT_QUANTUM_DECL void free_module(_In_ uintw mid) {
  META_LOCK_GUARD();

  if (mid >= module_results.size()) {
    std::cout << "Invalid argument: module ID not found!" << std::endl;
    meta_error = 2;
    return;
  }

  module_results[mid] = nullptr;
}

MICROSOFT_QUANTUM_DECL void forward(_In_ uintw mid, _In_ uintw dtype,
                                    _In_ uintw n, _In_reads_(n) uintw *shape,
                                    _In_reads_(n) uintw *stride,
                                    _In_ real1_s *d) {
  MODULE_LOCK_GUARD(mid);

  TensorPtr x;
  try {
    std::vector<tcapint> sh(n);
    std::vector<tcapint> st(n);
    std::transform(shape, shape + n, sh.begin(),
                   [](uintw x) { return (tcapint)x; });
    std::transform(stride, stride + n, st.begin(),
                   [](uintw x) { return (tcapint)x; });

    tcapint max_index = 0U;
    for (size_t i = 0U; i < sh.size(); ++i) {
      max_index += (sh[i] - 1U) * st[i];
    }
    if (!sh.empty()) {
      ++max_index;
    }

    if (dtype == 1U) {
      std::vector<real1> v(max_index);
      std::transform(d, d + max_index, v.begin(),
                     [](real1_s x) { return (real1)x; });
      x = std::make_shared<Tensor>(v, sh, st);
    } else {
      std::vector<complex> v(max_index);
      for (size_t i = 0U; i < max_index; ++i) {
        size_t j = i << 1U;
        v[i] = complex(d[j], d[j + 1U]);
      }
      x = std::make_shared<Tensor>(v, sh, st);
    }
  } catch (const std::exception &ex) {
    std::cout << ex.what() << std::endl;
    meta_error = 1;
  }

  try {
    module_results[mid]->t = module_results[mid]->m->forward(x);
  } catch (const std::exception &ex) {
    std::cout << ex.what() << std::endl;
    module_results[mid]->error = 1;
  }
}
}
