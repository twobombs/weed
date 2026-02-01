# Operation Implementations

This directory contains the source code for dispatching tensor operations to the appropriate kernels (CPU or GPU).

## Files

*   **abs.cpp**: Implementation of absolute value dispatch.
*   **clamp.cpp**: Implementation of clamp dispatch.
*   **commuting.cpp**: Implementation of addition and multiplication dispatch. It handles broadcasting and type promotion (Real + Complex -> Complex).
*   **div.cpp**: Implementation of division dispatch.
*   **in_place.cpp**: Implementation of in-place addition and subtraction dispatch.
*   **matmul.cpp**: Implementation of matrix multiplication dispatch.
*   **pow.cpp**: Implementation of power, exponentiation, and logarithm dispatch.
*   **real_unary.cpp**: Implementation of real-valued unary operations.
*   **reduce.cpp**: Implementation of reduction operations (sum, mean).
*   **sub.cpp**: Implementation of subtraction dispatch.
*   **sum.cpp**: Implementation of summation dispatch (often alias or specific reduction).
*   **unary.cpp**: Implementation of unary operations (ReLU, Sigmoid, Tanh) and their gradient computations.
