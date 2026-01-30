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

#include "storage/all_storage.hpp"
#include "tensors/complex_tensor.hpp"
#include "tensors/real_tensor.hpp"

#define GET_CONST_FLAT_TENSOR(type, i, o) const type *o = static_cast<const type *>(&i);
#define GET_FLAT_TENSOR(type, i, o) type *o = static_cast<type *>(&i);

#define CPU_INIT_2_SCALAR(ft, strg)                                            \
  const tcapint O_a = a.offset;                                                \
  const tcapint I_a = a.stride[0U];                                            \
  GET_CONST_FLAT_TENSOR(ft, a, pa);                                            \
  GET_STORAGE(strg, out, po);                                                  \
  size_t n = a.get_size()

#define CPU_INIT_2(ft, strg)                                                   \
  const tcapint O_a = a.offset;                                                \
  const tcapint I_a = a.stride[0U];                                            \
  const tcapint I_o = out.stride[0U];                                          \
  GET_CONST_FLAT_TENSOR(ft, a, pa);                                            \
  GET_STORAGE(strg, out, po);                                                  \
  size_t n = out.storage->size

#define CPU_INIT_2_IN_PLACE(ft1, ft2)                                          \
  const tcapint O_a = a.offset;                                                \
  const tcapint I_a = a.stride[0U];                                            \
  const tcapint O_b = b.offset;                                                \
  const tcapint I_b = b.stride[0U];                                            \
  GET_FLAT_TENSOR(ft1, a, pa);                                                 \
  GET_CONST_FLAT_TENSOR(ft2, b, pb);                                           \
  size_t n = a.get_size()

#define CPU_INIT_3(ft1, ft2, strg)                                             \
  const tcapint O_a = a.offset;                                                \
  const tcapint I_a = a.stride[0U];                                            \
  const tcapint O_b = b.offset;                                                \
  const tcapint I_b = b.stride[0U];                                            \
  const tcapint I_o = out.stride[0U];                                          \
  GET_CONST_FLAT_TENSOR(ft1, a, pa);                                           \
  GET_CONST_FLAT_TENSOR(ft2, b, pb);                                           \
  GET_STORAGE(strg, out, po);                                                  \
  size_t n = out.storage->size

#define CPU_GRAD_INIT_3(ft1, ft2, ft3)                                         \
  const tcapint O_d = din.offset;                                              \
  const tcapint I_d = din.stride[0U];                                          \
  const tcapint O_i = in.offset;                                               \
  const tcapint I_i = in.stride[0U];                                           \
  const tcapint O_o = dout.offset;                                             \
  const tcapint I_o = dout.stride[0U];                                         \
  GET_FLAT_TENSOR(ft1, din, pdi);                                              \
  GET_CONST_FLAT_TENSOR(ft2, in, pi);                                          \
  GET_CONST_FLAT_TENSOR(ft3, dout, po);                                        \
  size_t n = din.get_size()
