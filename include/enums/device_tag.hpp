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

#include "config.h"

namespace Weed {
/**
 * Back-end device types available in Weed
 */
enum DeviceTag { NONE_DEVICE = 0, DEFAULT_DEVICE = 1, CPU = 2, GPU = 3 };
} // namespace Weed
