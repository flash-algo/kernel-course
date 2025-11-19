# Scal Kernel

The `scal` operator scales the elements of a tensor by a given scalar factor.

## Mathematical Definition

Given a scalar `alpha` and an input tensor `y`, the kernel evaluates

$$
y = \alpha y
$$

The scaling is performed element by element, multiplying each entry of `y` by `alpha`.

## Kernel Implementations

- [Python Implementation](../kernel_course/python_ops/scal.py)
- [PyTorch Implementation](../kernel_course/pytorch_ops/scal.py)
- [Triton Implementation](../kernel_course/triton_ops/scal.py)
- [CuTe Implementation](../kernel_course/cute_ops/scal.py)

All backends share the interface:

```python
def scal(y: torch.Tensor, alpha: float) -> torch.Tensor:
    ...
```

## Testing

See the [test suite](../tests/test_scal.py) for the validation harness that exercises every backend.

```bash
pytest tests/test_scal.py -s
```