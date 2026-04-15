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

#include "modules/qwen.hpp"
#include "common/serializer.hpp"

namespace Weed {
void Qwen::save(std::ostream &os) const {
  Module::save(os);
  embed_tokens->save(os);
  layers->save(os);
  norm->save(os);
  lm_head->save(os);
}
} // namespace Weed