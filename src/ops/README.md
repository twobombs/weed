# Operation Implementations

This directory contains the source code for dispatching tensor operations to the appropriate kernels (CPU or GPU).

## Files

*   **abs.cpp**: Implementation of absolute value dispatch.
*   **clamp.cpp**: Implementation of clamp dispatch.
*   **commuting.cpp**: Implementation of addition and multiplication dispatch. It handles broadcasting and type promotion (Real + Complex -> Complex).
*   **copy_broadcast.cpp**: Implementation of broadcast index materialization.
*   **div.cpp**: Implementation of division dispatch.
*   **embedding.cpp**: Implementation of embedding lookup dispatch.
*   **in_place.cpp**: Implementation of in-place addition and subtraction dispatch.
*   **matmul.cpp**: Implementation of matrix multiplication dispatch.
*   **pow.cpp**: Implementation of power, exponentiation, and logarithm dispatch.
*   **real_extremum.cpp**: Implementation of max/min extrema dispatch.
*   **real_unary.cpp**: Implementation of real-valued unary operations.
*   **reduce.cpp**: Implementation of reduction operations (sum, mean).
*   **sub.cpp**: Implementation of subtraction dispatch.
*   **sum.cpp**: Implementation of summation dispatch.
*   **util.cpp**: Implementation of utility functions (e.g., validation).
