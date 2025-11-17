# Swap Kernel

The `swap` operator exchanges the contents of two tensors without mutating the original data.

## Mathematical Definition

Given input tensors `x` and `y`, the kernel evaluates

$$
x \leftrightarrow y
$$

The assignment is performed element by element, swapping the values between `x` and `y`.

## Kernel Implementations

- [Python Implementation](../kernel_course/python_ops/swap.py)
- [PyTorch Implementation](../kernel_course/pytorch_ops/swap.py)
- [Triton Implementation](../kernel_course/triton_ops/swap.py)
- [CuTe Implementation](../kernel_course/cute_ops/swap.py)

All backends share the interface:

```python
def swap(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
```

## Testing

See the [test suite](../tests/test_swap.py) for the validation harness that exercises every backend.

```bash
pytest tests/test_swap.py -s
```