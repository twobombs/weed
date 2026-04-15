# Tests

This directory contains the unit tests and benchmarks for the Weed library, built using the [Catch2](https://github.com/catchorg/Catch2) testing framework. The tests verify correctness of tensor operations, module implementations, and autograd functionality.

## Files

### [`test_main.cpp`](test_main.cpp)
Main entry point for the test runner.

**Purpose:**
- Parses command-line arguments
- Configures test environment
- Selects device (CPU/GPU) for testing
- Runs all test cases

**Usage:**


**Implementation:**


### [`tests.cpp`](tests.cpp)
Contains the actual test cases.

#### Tensor Storage Tests


#### Broadcasting Tests


#### Matrix Multiplication Tests


#### Activation Function Tests


#### Autograd Tests


#### Module Tests


#### Loss Function Tests


### [`tests.hpp`](tests.hpp)
Header file with common test utilities.

**Contents:**


### [`benchmarks.cpp`](benchmarks.cpp)
Performance benchmark tests.

#### Matrix Multiplication Benchmark


#### Forward Pass Benchmark


## Running Tests

### Build


### Execute


## Test Coverage

| Component | Tests |
|-----------|-------|
| Tensor Creation | ✓ |
| Tensor Arithmetic | ✓ |
| Broadcasting | ✓ |
| Matrix Multiplication | ✓ |
| Activation Functions | ✓ |
| Reduction Operations | ✓ |
| Autograd | ✓ |
| Linear Layer | ✓ |
| Sequential Model | ✓ |
| Loss Functions | ✓ |
| Device Transfer | ✓ |
| Serialization | ✓ |

## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

## Additional Files

### [`catch.hpp`](catch.hpp)
Catch2 testing framework header file used for C++ testing.
