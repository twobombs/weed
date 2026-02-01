# Tensor Operations

This directory contains the definitions and dispatch logic for tensor operations. It defines kernel structures that abstract the execution of operations across different devices (CPU/GPU) and data types (Real/Complex).

## Files

*   **abs.hpp**: Absolute value operation (`abs`).
*   **clamp.hpp**: Clamping operation (`clamp`), limiting values to a specified range.
*   **commuting.hpp**: Commutative binary operations (`add`, `mul`). Defines the `CommutingKernel` struct used to dispatch these operations.
*   **div.hpp**: Division operation (`div`).
*   **in_place.hpp**: In-place binary operations (`add_in_place`, `sub_in_place`). Defines the `InPlaceKernel` struct.
*   **matmul.hpp**: Matrix multiplication (`matmul`, `>>`, `<<`).
*   **pow.hpp**: Power (`pow`, `^`) and exponential (`exp`) operations. Also includes logarithm (`log`).
*   **real_unary.hpp**: Unary operations that might have specific real-valued behaviors or constraints.
*   **reduce.hpp**: Reduction operations base definitions.
*   **sub.hpp**: Subtraction operation (`sub`).
*   **sum.hpp**: Summation operation (`sum`, `mean`).
*   **unary.hpp**: Common unary operations like activation functions (`relu`, `sigmoid`, `tanh`) and their gradients. Defines the `UnaryKernel` struct.
