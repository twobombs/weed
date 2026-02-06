# CMake Modules

This directory contains CMake modules and scripts used to configure the build process of the Weed library.

## Files

*   **Boost.cmake**: Configures the Boost C++ libraries dependency. It defines `BOOST_AVAILABLE` if found.
*   **Complex_x2.cmake**: Likely configures double-precision complex number support or SIMD extensions.
*   **Coverage.cmake**: Sets up code coverage analysis (likely used with `ENABLE_CODECOVERAGE`).
*   **CppStd.cmake**: Ensures the compiler supports the required C++ standard (C++11 or later).
*   **EnvVars.cmake**: Helper for handling environment variables.
*   **Examples.cmake**: Configures the building of example executables.
*   **Format.cmake**: Configures `clang-format` targets for code formatting.
*   **FpMath.cmake**: Configures floating-point math optimizations (e.g., `-ffast-math`).
*   **OpenCL.cmake**: Handles the detection and configuration of OpenCL. It supports fetching headers for Apple/PPC, configures SnuCL if enabled, and adds custom commands to compile `.cl` kernels into headers.
*   **Pstridepow.cmake**: Configuration related to specific math optimizations or stride power functions (project-specific).
*   **Pthread.cmake**: Configures POSIX threads support.
*   **TCapPow.cmake**: Configuration for `tcapint` (tensor capacity integer) power functions.
