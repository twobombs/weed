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
#include "stddef.h"

#if defined(_WIN32) && !defined(__CYGWIN__)
#define MICROSOFT_QUANTUM_DECL __declspec(dllexport)
#define MICROSOFT_QUANTUM_DECL_IMPORT __declspec(dllimport)
#else
#define MICROSOFT_QUANTUM_DECL
#define MICROSOFT_QUANTUM_DECL_IMPORT
#endif

// SAL only defined in windows.
#ifndef _In_
#define _In_
#define _In_reads_(n)
#endif

typedef unsigned long long uintw;
typedef long long intw;

#if FPPOW < 6
typedef float real1_s;
#else
typedef double real1_s;
#endif

extern "C" {
// non-quantum
MICROSOFT_QUANTUM_DECL int get_error(_In_ const uintw mid);
MICROSOFT_QUANTUM_DECL uintw load_module(_In_ const char *f);
MICROSOFT_QUANTUM_DECL void free_module(_In_ uintw mid);
}
