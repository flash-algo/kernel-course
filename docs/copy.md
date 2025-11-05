# Copy Kernel

The `copy` operator duplicates a contiguous or strided vector into a destination tensor without mutating the source data.

## Mathematical Definition

Given an input tensor `x` and a destination tensor `y`, the kernel evaluates

$$
y = x
$$

The assignment is performed element by element and leaves `x` unchanged.

## Kernel Implementations

| Backend | Notes |
| --- | --- |
| [Python](../kernel_course/python_ops/copy.py) | Pure Python + PyTorch utilities provide the numerical ground truth. |
| [PyTorch](../kernel_course/pytorch_ops/copy.py) | Leverages native PyTorch routines for a quick implementation. |
| [Triton](../kernel_course/triton_ops/copy.py) | Demonstrates block scheduling plus masked loads and stores. |
| [CuTe](../kernel_course/cute_ops/copy.py) | Highlights tile composition, alignment, and shared-memory usage. |

All backends share the interface:

```python
def copy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	...
```

## Testing

See the [test suite](../tests/test_copy.py) for the validation harness that exercises every backend.

| Backend | Throughput |
| --- | --- |
| Python | n GB/s |
| PyTorch | n GB/s |
| Triton | n GB/s |
| CuTe | n GB/s |