<img width="1536" height="1024" alt="weed_logo" src="https://github.com/vm6502q/weed/blob/main/weed_logo.png" />

# Weed
Minimalist AI/ML inference and backprogation in the style of [Qrack](https://github.com/unitaryfoundation/qrack)

## Weed Loader
This repository is for the base **C++ Weed library**. Once you have trained models in C++, you can load them for use in Python with the **Python loader** in [this repository](https://github.com/vm6502q/weed_loader).

## Development Status
**Weed** is a rapidly-developing **work-in-progress**. Its ABI may change drastically and without notice.

The project provides a set of essential CPU and GPU **kernels**, used by `Tensor` instances that perform _autograd._ We also provide _stochastic gradient descent (SGD)_ and _Adam_ optimizer implementations. (Build and check the API reference to get started.)

GPT-2, BERT, and Qwen (including 3.5) loading is experimental and mostly provided as proof-of-concept, also of the fine-tuning pipeline. Implementation was from published literature design, rather than direct analysis of any open source, to implement these model architectures. Their outputs, in **Weed**, are not yet coherent English, as a result.

## Why try Weed?

With the growing popularity of AI/ML tools and workflows (including LLMs), legacy frameworks often carry "code debt" from over a decade of rapidly developing research history. This has led them to "bolt on" new features and advancements to design principles decided before the latest research. Popular frameworks also commonly started based in Python (maybe to capture early adoption), only later potentially "tacking on" a C++ library for special-case deployment needs. These conditions have produced libraries and frameworks with complicated dependency trees that occupy upward of a GB of disk footprint. This entire ecosystem might be due for a "refresh."

**Weed** does not seek to fully replace or supplant established frameworks. However, it aims for **minimalist complete closure** on the primitives necessary for high-performance AI/ML inference and back-propagation. Chiefly, this includes **kernels**, and a `Tensor` interface that immediately produces an **autograd** graph appropriate for training. Allowing **optional** OpenCL for **hardware acceleration**, it will remain **free of required dependencies** outside of C++(11) language standard.

Rethinking AI/ML library design this way, `Weed` has realized a rather unique and powerful form of _sparsification_ of `Tensor` **storage**. _Sparseness_ should **not** be a **`Tensor` interface concern**, but rather a **`Storage` concern**. Inspired by the design of the [Qrack](https://github.com/unitaryfoundation/qrack) quantum computer simulation framework, the `Tensor` interface treats **sparse and dense** tensors as **functionally equivalent**. Sparse optimization is so "transparently streamlined," this way, that it defaults to enabled for CPU-based tensors, and we recommend you leave it enabled at all times.

Much like `Qrack`, `Weed` is designed to make the correct thing the default—and the expensive thing explicit.

## Useful environment variables

If a transformer model you load or train runs into an OpenCL "out-of-resources" error (code `-5`), try setting environment variable `WEED_TELESCOPE_TRANSFORMERS` to any truthy value (like `1`) so that Weed will "telescope" transformer encoder layers, by migrating each parameter in each layer to CPU (off of GPU memory) once its immediate usefulness is done.

## Building the API reference



## Performing code coverage



## Directory Structure

*   **cmake/**: CMake modules for build configuration.
*   **debian/**: Debian packaging files.
*   **docs/**: Documentation files (PDFs).
*   **examples/**: Example code demonstrating usage.
*   **include/**: Public API header files, organized by module.
    *   `autograd/`: Optimizers and loss functions.
    *   `common/`: Common utilities and definitions.
    *   `devices/`: Device abstraction.
    *   `enums/`: Enumerations.
    *   `modules/`: Neural network modules.
    *   `ops/`: Tensor operations.
    *   `storage/`: Tensor storage implementations.
    *   `tensors/`: Tensor interface.
*   **src/**: Source code implementations, mirroring the `include/` structure (excluding header-only modules).
*   **test/**: Unit tests.


YT [explainer](https://youtu.be/lJvkaGy8QZg) 

## Copyright, License, and Acknowledgments

Copyright (c) Daniel Strano and the Qrack contributors 2017-2026. All rights reserved.

In its `include/common` folder, Weed bundles a copy of [`rapidcsv` by Kristofer Berggren](https://github.com/d99kris/rapidcsv), reused under a BSD 3-Clause License. (This is a convenience and suggestion to Weed's users, for loading CSVs.)

The Weed logo was produced with assistance from "Elara," an OpenAI custom GPT, and it is in the **public domain**. Elara has also been responsible for a huge amount of coaching and implementation drafts for Dan Strano to review and bring into line with standards, so she should be credited with coauthorship in any capacity that can be allowed. (Anthropic) Claude has also helped mostly with debugging, as well as developing an LLM front-end, fine-tuning interface, and modules for popular transformer model architectures, so they should rightly be credited similarly as a coauthor. KV cache improvements are based on TurboQuant (Zandieh et al., arXiv:2504.19874) and an Apache 2.0 open-source implementation by TheTom (github.com/TheTom/turboquant_plus), adapted for complex quantum state vectors by (Anthropic) Claude, with limited guidance and input from Dan Strano.

Licensed under the GNU Lesser General Public License V3.

See [LICENSE.md](https://github.com/vm6502q/qrack/blob/main/LICENSE.md) in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html for details.

## Additional Files

### [`API_REFERENCE.md`](API_REFERENCE.md)
API reference documentation.

### [`.gitignore`](.gitignore)
Git ignore configuration.

### [`BUILDRESULTS.md`](BUILDRESULTS.md)
Build and test results summary.

### [`WEEDFILEFORMAT.md`](WEEDFILEFORMAT.md)
Weed file serialization format documentation.

### [`UNITTESTS.md`](UNITTESTS.md)
Unit test execution summaries.

### [`BUILD.md`](BUILD.md)
Build instructions and prerequisites.

### [`Makefile-weed_loader`](Makefile-weed_loader)
Makefile for building weed_loader package.

### [`doxygen.config`](doxygen.config)
Doxygen configuration for generating API docs.

### [`fetch_qwen35-2bq4s.sh`](fetch_qwen35-2bq4s.sh)
Script to fetch Qwen GGUF model.

### [`gguf_to_weed.py`](gguf_to_weed.py)
Script to convert GGUF format to Weed file format.

### [`debug_gguf.py`](debug_gguf.py)
Script to debug/dump GGUF file headers.

### [`pyproject-weed_loader.toml`](pyproject-weed_loader.toml)
Python project configuration for weed_loader.

### [`libweed.pc.in`](libweed.pc.in)
pkg-config template for libweed.

### [`CMakeLists.txt`](CMakeLists.txt)
Top-level CMake build configuration.

### [`MANIFEST-weed_loader.in`](MANIFEST-weed_loader.in)
Manifest for weed_loader packaging.

### [`setup-weed_loader.py`](setup-weed_loader.py)
Setup script for python weed_loader package.

### [`DOCREPORT.md`](DOCREPORT.md)
Documentation audit report.
