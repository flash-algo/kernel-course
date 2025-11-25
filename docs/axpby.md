# AXPBY Kernel

The `axpby` operator computes a linear combination of two input tensors, scaling each by a specified scalar factor, and stores the result in a destination tensor.

## Mathematical Definition

Given input tensors `x` and `y`, scalar coefficients `a` and `b`, the kernel evaluates

$$
y = a \times x + b \times y
$$

The operation is performed element by element, combining corresponding entries from `x` and `y` according to the specified scaling factors.

## Kernel Implementations

- [Python Implementation](../kernel_course/python_ops/axpby.py)
- [PyTorch Implementation](../kernel_course/pytorch_ops/axpby.py)
- [Triton Implementation](../kernel_course/triton_ops/axpby.py)
- [CuTe Implementation](../kernel_course/cute_ops/axpby.py)

All backends share the interface:

```python
def axpby(x: torch.Tensor, y: torch.Tensor, a: float, b: float) -> torch.Tensor:
    ...
```

## Testing

See the [test suite](../tests/test_axpby.py) for the validation harness that exercises every backend.

```bash
pytest tests/test_axpby.py -s
```
