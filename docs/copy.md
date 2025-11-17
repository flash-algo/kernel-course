# Copy Kernel

The `copy` operator duplicates a contiguous or strided vector into a destination tensor without mutating the source data.

## Mathematical Definition

Given an input tensor `x` and a destination tensor `y`, the kernel evaluates

$$
y = x
$$

The assignment is performed element by element and leaves `x` unchanged.

## Kernel Implementations

- [Python Implementation](../kernel_course/python_ops/copy.py)
- [PyTorch Implementation](../kernel_course/pytorch_ops/copy.py)
- [Triton Implementation](../kernel_course/triton_ops/copy.py)
- [CuTe Implementation](../kernel_course/cute_ops/copy.py)

All backends share the interface:

```python
def copy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	...
```

## Testing

See the [test suite](../tests/test_copy.py) for the validation harness that exercises every backend.

```bash
pytest tests/test_copy.py -s
```
