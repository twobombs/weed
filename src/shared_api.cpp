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

#include <fstream>
#include <iostream>
#include <mutex>

#define META_LOCK_GUARD()                                                      \
  const std::lock_guard<std::mutex> meta_lock(meta_operation_mutex)

// MODULE_LOCK_GUARD variants will lock module_mutexes[nullptr], if the
// requested module doesn't exist. This is CORRECT behavior. This will
// effectively emplace a mutex for nullptr key.
#if CPP_STD > 13
#define MODULE_LOCK_GUARD(module)                                              \
  std::unique_ptr<const std::lock_guard<std::mutex>> module_lock;              \
  if (true) {                                                                  \
    std::lock(meta_operation_mutex, module_mutexes[module]);                   \
    const std::lock_guard<std::mutex> meta_lock(meta_operation_mutex,          \
                                                std::adopt_lock);              \
    module_lock = std::make_unique<const std::lock_guard<std::mutex>>(         \
        module_mutexes[module], std::adopt_lock);                              \
  }
#else
#define MODULE_LOCK_GUARD(module)                                                                                  \
  std::unique_ptr<const std::lock_guard<std::mutex>> module_lock;                                                  \
  if (true) {                                                                                                      \
    std::lock(meta_operation_mutex, module_mutexes[module]);                                                       \
    const std::lock_guard<std::mutex> meta_lock(meta_operation_mutex,                                              \
                                                std::adopt_lock);CMPLX_DEFAULT_ARG, false, true, hp, -1, true, sp)                               \
        module_lock = std::unique_ptr<const std::lock_guard<std::mutex>>(                                            \
            new const std::lock_guard<std::mutex>(module_mutexes[module], std::adopt_lock));                       \
  }
#endif

#define MODULE_LOCK_GUARD_VOID(mid)                                            \
  if (mid >= modules.size()) {                                                 \
    std::cout << "Invalid argument: module ID not found!" << std::endl;        \
    meta_error = 2;                                                            \
    return;                                                                    \
  }                                                                            \
  QInterfacePtr module = modules[mid];                                         \
  MODULE_LOCK_GUARD(module.get())                                              \
  if (!module) {                                                               \
    return;                                                                    \
  }

#define MODULE_LOCK_GUARD_TYPED(mid, def)                                      \
  if (mid >= modules.size()) {                                                 \
    std::cout << "Invalid argument: module ID not found!" << std::endl;        \
    meta_error = 2;                                                            \
    return def;                                                                \
  }                                                                            \
                                                                               \
  QInterfacePtr module = modules[mid];                                         \
  MODULE_LOCK_GUARD(module.get())                                              \
  if (!module) {                                                               \
    return def;                                                                \
  }

#define MODULE_LOCK_GUARD_BOOL(mid) MODULE_LOCK_GUARD_TYPED(mid, false)

#define MODULE_LOCK_GUARD_DOUBLE(mid) MODULE_LOCK_GUARD_TYPED(mid, 0.0)

#define MODULE_LOCK_GUARD_INT(mid) MODULE_LOCK_GUARD_TYPED(mid, 0U)

using namespace Weed;

std::mutex meta_operation_mutex;
int meta_error = 0;

std::vector<int> module_errors;
std::vector<ModulePtr> modules;

std::vector<int> tensor_errors;
std::vector<BaseTensorPtr> tensors;

void _darray_to_creal1_array(double *params, tcapint componentCount,
                             complex *amps) {
  for (tcapint j = 0U; j < componentCount; ++j) {
    amps[j] = complex(real1(params[2U * j]), real1(params[2U * j + 1U]));
  }
}

extern "C" {

/**
 * (External API) Poll after each operation to check whether error occurred.
 */
MICROSOFT_QUANTUM_DECL int get_error(_In_ const uintw mid) {
  if (meta_error) {
    meta_error = 0;
    return 2;
  }

  return module_errors[mid];
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
    while ((id < modules.size()) && modules[id]) {
      ++id;
    }
    if (id == modules.size()) {
      modules.push_back(m);
    } else {
      modules[id] = m;
    }
  }

  return id;
}
MICROSOFT_QUANTUM_DECL void free_module(_In_ uintw mid) {
  META_LOCK_GUARD();

  if (mid >= modules.size()) {
    std::cout << "Invalid argument: module ID not found!" << std::endl;
    meta_error = 2;
    return;
  }

  modules[mid] = nullptr;
}
}
