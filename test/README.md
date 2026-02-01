# Tests

This directory contains the unit tests for the Weed library, built using the [Catch2](https://github.com/catchorg/Catch2) framework.

## Files

*   **test_main.cpp**: The main entry point for the test runner. It parses command-line arguments to configure the test environment, such as selecting the device (CPU/GPU) to run tests on.
*   **tests.cpp**: Contains the actual test cases covering:
    *   Tensor storage and device transfers.
    *   Arithmetic operations (add, sub, mul, div).
    *   Activation functions (ReLU, Sigmoid, Tanh).
    *   Matrix multiplication.
    *   Autograd functionality (verifying gradients).
*   **tests.hpp**: Header file defining common macros, globals (like the current `DeviceTag` being tested), and includes.
*   **catch.hpp**: The single-header Catch2 library.

## Running Tests

Tests are typically built and run via CMake:

```sh
cd _build
make unittest
./unittest
```

You can select specific devices using command line flags:
*   `--device-cpu`: Run CPU tests.
*   `--device-gpu`: Run GPU tests.
