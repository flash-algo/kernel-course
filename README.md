<div align="center">
  <img src="./assets/banner.png" alt="banner" width="100%">
</div>

Learn how to develop high-performance kernels with [PyTorch](https://github.com/pytorch/pytorch), [Triton](https://github.com/triton-lang/triton), and [CuTe](https://github.com/NVIDIA/cutlass) while preserving numerical equivalence with the Python reference implementations. The exercises emphasize translating clear Python prototypes into optimized GPU kernels without sacrificing correctness.

## Basic Linear Algebra Subprograms

The following BLAS kernels have been implemented in multiple frameworks. For each kernel, a ✅ indicates that the implementation is complete and verified to be numerically equivalent to the Python reference, a ❌ indicates that the implementation is pending. For more details on each kernel, please **click the name or icon**.


| Name | Description | Equation | Flops | Data | Python | PyTorch | Triton | CuTe | Test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| axpy | update vector| $ y = y + \alpha x $ | $ 2n $ | $ 2n $ | ❌ | ❌ | ❌ | ❌ | ❌ |
| scal | scale vector | $ y = \alpha y $ | $ n $ | $ n $ | ❌ | ❌ | ❌ | ❌ | ❌ |
| copy | copy vector | $ y = x $ | $ 0 $ | $ 2n $ | ❌ | ❌ | ❌ | ❌ | ❌ |
| swap | swap vectors | $ x \leftrightarrow y $ | $ 0 $ | $ 2n $ | ❌ | ❌ | ❌ | ❌ | ❌ |
| dot | dot product | $ = x^\top y $ | $ 2n $ | $ 2n $ | ❌ | ❌ | ❌ | ❌ | ❌ |
| gemv | general matrix-vector multiply | $ y = \alpha A x + \beta y $ | $ 2mn $ | $ mn + m + n $ | ❌ | ❌ | ❌ | ❌ | ❌ |
| ger | general rank-1 update | $ A = A + \alpha x y^\top $ | $ 2mn $ | $ mn $ | ❌ | ❌ | ❌ | ❌ | ❌ |
| gemm | general matrix-matrix multiply | $ C = \alpha A B + \beta C $ | $ 2mnk $ | $ mk + nk + mn $ | ❌ | ❌ | ❌ | ❌ | ❌ |