# Building Weed AI/ML Library

This document provides detailed instructions for building the Weed library from source.

## Prerequisites

### Required Tools

| Tool | Minimum Version | Notes |
|------|-----------------|-------|
| CMake | 3.10+ | Build system |
| C++ Compiler | C++11+ | GCC 4.8+, Clang 3.3+, MSVC 2015+ |
| Make | Any | For build execution |
| xxd | Any | For OpenCL kernel precompilation |

### Optional Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| OpenCL SDK | GPU acceleration | See [OpenCL Setup](#opencl-setup) |
| Qrack library | Quantum computing support | See [Qrack Setup](#qrack-setup) |
| pthread | Multi-threading | Usually included with system |
| libquadmath | Float128 support | `libquadmath-dev` (Debian/Ubuntu) |

## Quick Start

### Standard Build (CPU only)

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build
make -j$(nproc)

# Install (optional, requires sudo)
sudo make install
```

### Build with OpenCL (GPU acceleration)

```bash
mkdir build && cd build

# Configure with OpenCL enabled
cmake .. -DWEED_ENABLE_OPENCL=ON

# Build
make -j$(nproc)
```

### Build without OpenCL

```bash
mkdir build && cd build

# Configure with OpenCL disabled
cmake .. -DWEED_ENABLE_OPENCL=OFF

# Build
make -j$(nproc)
```

## Build Configuration Options

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `WEED_ENABLE_OPENCL` | ON | Enable OpenCL GPU acceleration |
| `WEED_ENABLE_QRACK` | ON | Enable Qrack quantum computing support |
| `WEED_ENABLE_SNUCL` | OFF | Enable SnuCL cluster support |
| `WEED_ENABLE_PTHREAD` | ON | Enable POSIX threads |
| `ENABLE_EXAMPLES` | ON | Build example programs |
| `ENABLE_TESTS` | ON | Build test suite and benchmarks |
| `WEED_ENABLE_OOO_OCL` | ON | Enable OpenCL out-of-order queue (v2.0) |
| `PACK_DEBIAN` | OFF | Build Debian package |

### Precision Configuration

| Option | Default | Values | Description |
|--------|---------|--------|-------------|
| `WEED_FPPOW` | 5 | 4-7 | Floating-point precision (4=half, 5=float, 6=double, 7=float128) |
| `WEED_TCAPPOW` | 5 | 3-7 | Tensor capacity power (3=8 qubits, 7=128 qubits) |
| `WEED_CPP_STD` | 14 | 11, 14, 17, 20, 23 | C++ standard version |

### Example Build Commands

```bash
# Double precision build
cmake .. -DWEED_FPPOW=6

# Maximum tensor capacity (128 qubits)
cmake .. -DWEED_TCAPPOW=7

# C++17 standard
cmake .. -DWEED_CPP_STD=17

# Float128 precision (requires libquadmath)
cmake .. -DWEED_FPPOW=7 -DWEED_ENABLE_QRACK=OFF

# Build without examples and tests
cmake .. -DENABLE_EXAMPLES=OFF -DENABLE_TESTS=OFF

# Build with specific C++ standard and precision
cmake .. -DWEED_CPP_STD=17 -DWEED_FPPOW=6 -DWEED_TCAPPOW=6
```

## OpenCL Setup

### Linux (Ubuntu/Debian)

```bash
# Install OpenCL ICD loader and headers
sudo apt-get install ocl-icd-opencl-dev opencl-headers

# Install NVIDIA OpenCL (for NVIDIA GPUs)
sudo apt-get install nvidia-opencl-dev

# Install AMD OpenCL (for AMD GPUs)
# Download from: https://www.amd.com/en/support/software/amp-software-development-kit
# Or use Intel oneAPI
sudo apt-get install intel-opencl-icd
```

### Linux (AMD/Intel SDK)

```bash
# Set AMD SDK path if not in default location
cmake .. -DOPENCL_AMDSDK=/opt/AMDAPPSDK-3.0
```

### macOS

```bash
# OpenCL is included with macOS SDK
# No additional installation required

# If using Apple Silicon, ensure OpenCL headers are available
# The build system will fetch them automatically
```

### Windows

```powershell
# Install AMD APP SDK or Intel oneAPI
# Set the SDK path
cmake .. -DOPENCL_AMDSDK="C:/Program Files (x86)/Common Files/Intel/Shared Libraries"
```

## Qrack Setup

Qrack is a quantum computing simulator library.

### Installation

```bash
# Clone Qrack repository
git clone https://github.com/QuantinuumHQ/qrack.git
cd qrack

# Build Qrack
mkdir build && cd build
cmake .. -DBUILD_QFT=ON -DBUILD_QRACK=ON
make -j$(nproc)
sudo make install
```

### Build with Qrack

```bash
# Default installation path
cmake .. -DQRACK_DIR="/usr/local/lib/qrack" -DQRACK_INCLUDE="/usr/local/include"

# Custom installation path
cmake .. -DQRACK_DIR="/opt/qrack/lib" -DQRACK_INCLUDE="/opt/qrack/include"
```

### Disable Qrack

```bash
cmake .. -DWEED_ENABLE_QRACK=OFF
```

## Platform-Specific Instructions

### Linux (Ubuntu/Debian)

```bash
# Install all dependencies
sudo apt-get update
sudo apt-get install -y \
    cmake \
    g++ \
    make \
    opencl-headers \
    ocl-icd-opencl-dev \
    libquadmath-dev \
    xxd

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Linux (RHEL/CentOS/Fedora)

```bash
# Install dependencies
sudo dnf install -y \
    cmake \
    gcc-c++ \
    make \
    opencl-headers \
    ocl-icd-devel \
    libquadmath-devel \
    xxd

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### macOS

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install CMake (if not already installed)
brew install cmake

# Build
mkdir build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)
```

### Windows (MSVC)

```powershell
# Install Visual Studio with C++ workload
# Install CMake from https://cmake.org/download/

# Open Developer Command Prompt
cmake -G "Visual Studio 17 2022" -A x64 .
cmake --build . --config Release
```

### Windows (MinGW)

```powershell
# Install MinGW-w64 and CMake
# Add to PATH: C:\msys64\mingw64\bin

# Build
mkdir build && cd build
cmake -G "MinGW Makefiles" ..
make -j$(nproc)
```

### Emscripten (WebAssembly)

```bash
# Install Emscripten
source /path/to/emsdk/emsdk_env.sh

# Build for WebAssembly
mkdir build && cd build
emcmake cmake .. -DWEED_ENABLE_PTHREAD=ON
emmake make -j$(nproc)
```

## Build Targets

### Library Targets

| Target | Type | Description |
|--------|------|-------------|
| `weed` | Static | Main Weed library |
| `weed_shared` | Shared | Shared library version |
| `weed_cl_precompile` | Executable | OpenCL kernel precompiler |

### Test Targets

| Target | Description |
|--------|-------------|
| `unittest` | Unit test suite |
| `benchmarks` | Performance benchmarks |

### Example Targets

| Target | Description |
|--------|-------------|
| `xor` | XOR gate example |
| `xor_qrack` | XOR with Qrack |
| `heart_attack` | Medical prediction example |
| `quantum_volume` | Quantum volume benchmark |
| `binary_addition_transformer` | Transformer example |

## Running Tests

```bash
# Run all tests
./unittest

# Run specific test suite
./unittest "[tensor]"

# Show test descriptions
./unittest --list-tests

# Run with output
./unittest --reporters=console
```

## Running Benchmarks

```bash
# Run benchmarks
./benchmarks

# Run specific benchmark
./benchmarks "[matmul]"
```

## Running Examples

```bash
# Build examples first
cmake .. -DENABLE_EXAMPLES=ON

# Run examples
./xor
./heart_attack
./quantum_volume
```

## Installation

### Standard Installation

```bash
# After building
sudo make install

# Installs to:
# /usr/local/lib/libweed.a
# /usr/local/include/weed/
# /usr/local/lib/pkgconfig/libweed.pc
```

### Custom Installation Path

```bash
cmake .. -DCMAKE_INSTALL_PREFIX=/opt/weed
make
sudo make install
```

### Uninstallation

```bash
sudo xargs rm < install_manifest.txt
```

## Build Verification

### Check Build Output

```bash
# Verify configuration
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON

# Check for warnings
make 2>&1 | grep -i warning
```

### Verify Library

```bash
# Check library exists
ls -la build/libweed.a

# Check symbols
nm -D build/libweed.a | head -20
```

## Troubleshooting

### OpenCL Not Found

```bash
# Check OpenCL installation
clinfo

# Verify library
ldconfig -p | grep opencl

# Manual path specification
cmake .. -DOpenCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenCL.so
```

### Qrack Not Found

```bash
# Check Qrack installation
ls /usr/local/lib/libqrack*

# Specify custom path
cmake .. -DQRACK_DIR="/opt/qrack" -DQRACK_INCLUDE="/opt/qrack/include"
```

### pthread Issues

```bash
# Check pthread availability
ldconfig -p | grep pthread

# Force pthread enable
cmake .. -DWEED_ENABLE_PTHREAD=ON
```

### Compiler Errors

```bash
# Check compiler version
g++ --version

# Ensure C++ standard support
cmake .. -DWEED_CPP_STD=17
```

## Performance Tips

### Enable Fast Math

```bash
# The build system enables -O3 by default
# For additional optimization, edit CMakeLists.txt
```

### Thread Configuration

```bash
# Set thread count
export OMP_NUM_THREADS=4
```

### Memory Optimization

```bash
# Adjust stride power for parallel loops
cmake .. -DWEED_PSTRIDEPOW=10
```

## Clean Build

```bash
# Remove build directory
rm -rf build

# Or clean in-place
make clean
cmake ..
```

## License

This project is licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0). See [LICENSE.md](LICENSE.md) for details.

## References

- [CMakeLists.txt](CMakeLists.txt) - Main build configuration
- [cmake/](cmake/) - CMake modules
- [README.md](README.md) - Project overview
- [API_REFERENCE.md](API_REFERENCE.md) - API documentation
