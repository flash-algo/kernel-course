# GEMM Kernel

The `gemm` operator computes the matrix-matrix product of two matrices.

## Mathematical Definition

Given input matrices `A` and `B`, along with an output matrix `C` and scalars `α` and `β`, the kernel evaluates

$$
C = \alpha A B + \beta C
$$

The matrix-matrix product is computed by multiplying the matrix `A` with the matrix `B`, scaling the result by `α`, scaling the matrix `C` by `β`, and then adding the two scaled results together to produce the updated matrix `C`.

## Kernel Implementations

- [Python Implementation](../kernel_course/python_ops/gemm.py)
- [PyTorch Implementation](../kernel_course/pytorch_ops/gemm.py)
- [Triton Implementation](../kernel_course/triton_ops/gemm.py)
- [CuTe Implementation](../kernel_course/cute_ops/gemm.py)

All backends share the interface:

```python
def gemm(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    ...
```

## Testing

See the [test suite](../tests/test_gemm.py) for the validation harness that exercises every backend.

```bash
pytest tests/test_gemm.py -s
```
