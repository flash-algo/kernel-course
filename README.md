<div align="center">
  <img src="./assets/banner.png" alt="banner" width="100%">
</div>

Learn how to develop high-performance kernels with [PyTorch](https://github.com/pytorch/pytorch), [Triton](https://github.com/triton-lang/triton), and [CuTe](https://github.com/NVIDIA/cutlass) while preserving numerical equivalence with the Python reference implementations. The exercises emphasize translating clear Python prototypes into optimized GPU kernels without sacrificing correctness.

## Basic Linear Algebra Subprograms

The following BLAS kernels have been implemented in multiple frameworks. For each kernel, a ✅ indicates that the implementation is complete and verified to be numerically equivalent to the Python reference. A ❌ indicates that the implementation is pending. For more details on each kernel, please **click the name or icon**.


| kernel | Python | PyTorch | Triton | CuTe | Test |
| --- | --- | --- | --- | --- | --- |
| [add](./docs/add.md) | [✅](./kernel_course/python_ops/add.py) | [✅](./kernel_course/torch_ops/add.py) | [✅](./kernel_course/triton_ops/add.py) | [❌](./kernel_course/cute_ops/add.py) | [✅](./tests/test_add.py) |


