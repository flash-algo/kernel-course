# Dot Kernel

The `dot` operator computes the dot product of two vectors.

## Mathematical Definition

Given two input vectors `x` and `y`, the kernel evaluates

$$
z = x^\top y
$$

The dot product is computed by multiplying corresponding elements of `x` and `y`, and summing the results to produce a single scalar value `z`.

## Kernel Implementations

- [Python Implementation](../kernel_course/python_ops/dot.py)
- [PyTorch Implementation](../kernel_course/pytorch_ops/dot.py)
- [Triton Implementation](../kernel_course/triton_ops/dot.py)
- [CuTe Implementation](../kernel_course/cute_ops/dot.py)

All backends share the interface:

```python
def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    ...
```

## Testing

See the [test suite](../tests/test_dot.py) for the validation harness that exercises every backend.

```bash
pytest tests/test_dot.py -s
```
