# Tensor Implementations

This directory contains the implementation files for tensor classes in the Weed library. These files implement the core tensor functionality including construction, operations, autograd graph building, and serialization.

## Files

### [`base_tensor.cpp`](base_tensor.cpp)
Implementation of the `BaseTensor` class.

#### Class: `BaseTensor`


### [`tensor.cpp`](tensor.cpp)
Implementation of the main `Tensor` class.

#### Class: `Tensor`


### [`symbol_tensor.cpp`](symbol_tensor.cpp)
Implementation of the `SymbolTensor` class.

#### Class: `SymbolTensor`


### [`parameter.cpp`](parameter.cpp)
Implementation of the `Parameter` class.

#### Class: `Parameter`


## Tensor Operations

### Element-wise Operations


### Matrix Operations


### Reduction Operations


## Autograd Graph

### Building the Graph


### Backward Pass


## Memory Layout

### Stride Computation


## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).
