# GEMV Kernel

The `gemv` operator computes the matrix-vector product of a matrix and a vector.

## Mathematical Definition

Given an input matrix `A` and input vectors `x` and `y`, along with scalars `α` and `β`, the kernel evaluates

$$
y = \alpha A x + \beta y
$$

The matrix-vector product is computed by multiplying the matrix `A` with the vector `x`, scaling the result by `α`, scaling the vector `y` by `β`, and then adding the two scaled results together to produce the updated vector `y`.

## Kernel Implementations

- [Python Implementation](../kernel_course/python_ops/gemv.py)
- [PyTorch Implementation](../kernel_course/pytorch_ops/gemv.py)
- [Triton Implementation](../kernel_course/triton_ops/gemv.py)
- [CuTe Implementation](../kernel_course/cute_ops/gemv.py)

All backends share the interface:

```python
def gemv(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    ...
```

## Testing

See the [test suite](../tests/test_gemv.py) for the validation harness that exercises every backend.

```bash
pytest tests/test_gemv.py -s
```
