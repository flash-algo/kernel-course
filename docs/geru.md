# GERU Kernel

The `geru` operator computes the outer product of two vectors and adds the result to a matrix.

## Mathematical Definition

Given an input matrix `A` and input vectors `x` and `y`, along with a scalar `α`, the kernel evaluates

$$
A = A + \alpha x y^\top
$$

The outer product is computed by multiplying the vector `x` with the transpose of vector `y`, scaling the result by `α`, and then adding it to the matrix `A` to produce the updated matrix `A`.

## Kernel Implementations

- [Python Implementation](../kernel_course/python_ops/geru.py)
- [PyTorch Implementation](../kernel_course/pytorch_ops/geru.py)
- [Triton Implementation](../kernel_course/triton_ops/geru.py)
- [CuTe Implementation](../kernel_course/cute_ops/geru.py)

All backends share the interface:

```python
def geru(A: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    ...
```

## Testing

See the [test suite](../tests/test_geru.py) for the validation harness that exercises every backend.

```bash
pytest tests/test_geru.py -s
```