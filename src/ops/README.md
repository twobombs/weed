# Operation Implementations

This directory contains the implementation files for tensor operations in the Weed library. These files implement the dispatch logic that routes operations to the appropriate CPU or GPU kernels.

## Files

### [`util.cpp`](util.cpp)
Implementation of utility functions for tensor operation validation.

**Implementation Details:**


### [`commuting.cpp`](commuting.cpp)
Implementation of addition and multiplication dispatch.

#### Function: `add_dispatch`


#### Function: `mul_dispatch`


### [`sub.cpp`](sub.cpp)
Implementation of subtraction dispatch.

#### Function: `sub_dispatch`


### [`div.cpp`](div.cpp)
Implementation of division dispatch.

#### Function: `div_dispatch`


### [`in_place.cpp`](in_place.cpp)
Implementation of in-place operations.

#### Function: `add_in_place_dispatch`


### [`matmul.cpp`](matmul.cpp)
Implementation of matrix multiplication dispatch.

#### Function: `matmul_dispatch`


### [`sum.cpp`](sum.cpp)
Implementation of summation dispatch.

#### Function: `sum_dispatch`


#### Function: `mean_dispatch`


### [`reduce.cpp`](reduce.cpp)
Implementation of reduction operations.

#### Function: `max_dispatch`


### [`real_extremum.cpp`](real_extremum.cpp)
Implementation of real-valued extrema operations.

#### Function: `real_max_dispatch`


### [`real_unary.cpp`](real_unary.cpp)
Implementation of real-valued unary operations.

#### Function: `abs_dispatch`


#### Function: `relu_dispatch`


#### Function: `sigmoid_dispatch`


### [`pow.cpp`](pow.cpp)
Implementation of power, exponential, and logarithm operations.

#### Function: `pow_dispatch`


#### Function: `exp_dispatch`


#### Function: `log_dispatch`


### [`copy_broadcast.cpp`](copy_broadcast.cpp)
Implementation of broadcast index materialization.

#### Function: `copy_broadcast_dispatch`


### [`embedding.cpp`](embedding.cpp)
Implementation of embedding lookup dispatch.

#### Function: `embedding_dispatch`


### [`triu_fill.cpp`](triu_fill.cpp)
Implementation of upper triangular fill operation.

#### Function: `triu_fill_dispatch`


## Dispatch Pattern



## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

## Additional Files

### [`softmax.cpp`](softmax.cpp)
Implementation of Softmax activation operation.

### [`abs.cpp`](abs.cpp)
Implementation of absolute value operation.

### [`logsoftmax.cpp`](logsoftmax.cpp)
Implementation of Log-Softmax activation operation.

### [`clamp.cpp`](clamp.cpp)
Implementation of value clamping operation.
