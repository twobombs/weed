# Storage Implementations

This directory contains the implementation files for storage classes in the Weed library. These files handle memory allocation, device transfers, and provide the concrete implementations for different storage backends (CPU/GPU, Dense/Sparse, Real/Complex).

## Files

### [`storage.cpp`](storage.cpp)
Implementation of the base `Storage` class.

#### Class: `Storage`


### [`cpu_real_storage.cpp`](cpu_real_storage.cpp)
Implementation of dense real number storage for CPU.

#### Class: `CpuRealStorage`


### [`cpu_complex_storage.cpp`](cpu_complex_storage.cpp)
Implementation of dense complex number storage for CPU.

#### Class: `CpuComplexStorage`


### [`cpu_int_storage.cpp`](cpu_int_storage.cpp)
Implementation of dense integer storage for CPU.

#### Class: `CpuIntStorage`


### [`sparse_cpu_real_storage.cpp`](sparse_cpu_real_storage.cpp)
Implementation of sparse real number storage for CPU.

#### Class: `SparseCpuRealStorage`


### [`sparse_cpu_complex_storage.cpp`](sparse_cpu_complex_storage.cpp)
Implementation of sparse complex number storage for CPU.

#### Class: `SparseCpuComplexStorage`


### [`gpu_real_storage.cpp`](gpu_real_storage.cpp)
Implementation of dense real number storage for GPU.

#### Class: `GpuRealStorage`


### [`gpu_complex_storage.cpp`](gpu_complex_storage.cpp)
Implementation of dense complex number storage for GPU.

#### Class: `GpuComplexStorage`


### [`gpu_int_storage.cpp`](gpu_int_storage.cpp)
Implementation of dense integer storage for GPU.

#### Class: `GpuIntStorage`


### `TypedStorage`
Implementation of the `TypedStorage` template class.

#### Class Template: `TypedStorage<T>`


### `CpuStorage`
Implementation of the `CpuStorage` template class.

#### Class Template: `CpuStorage<T>`


### `GpuStorage`
Implementation of the `GpuStorage` template class.

#### Class Template: `GpuStorage<T>`


### `SparseCpuStorage`
Implementation of the `SparseCpuStorage` template class.

#### Class Template: `SparseCpuStorage<T>`


## Storage Factory Functions



## Memory Management

### Allocation Pattern


### Migration Pattern


## Sparse Storage Efficiency

### Memory Usage
| Storage Type | Memory Formula |
|--------------|----------------|
| Dense Real | `size * sizeof(real1)` |
| Dense Complex | `size * sizeof(complex)` |
| Sparse Real | `sparse_size * sizeof(real1) + sparse_size * sizeof(tcapint)` |

### When to Use Sparse
- >90% zero elements
- Large tensors with few non-zero values
- Embedding lookups with padding

## License

Licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).
