import pytest
import torch

from kernel_course import testing
from kernel_course.python_ops import scal as python_scal

try:
    from kernel_course.pytorch_ops import scal as pytorch_scal

    HAS_PYTORCH = True
except Exception:
    pytorch_scal = None
    HAS_PYTORCH = False

try:
    from kernel_course.triton_ops import scal as triton_scal

    HAS_TRITON = True
except Exception:
    triton_scal = None
    HAS_TRITON = False

try:
    from kernel_course.cute_ops import scal as cute_scal

    HAS_CUTE = True
except Exception:
    cute_scal = None
    HAS_CUTE = False


def factory(
    numel: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    y = torch.linspace(0.0, 1.0, steps=numel, device=device, dtype=dtype)
    alpha = 3.14
    return (y, alpha), {}


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="requires MPS"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
)
@pytest.mark.parametrize(
    "numel",
    [1 << 4, 1 << 8, 1 << 16],
)
def test_scal_benchmark(device: torch.device, dtype: torch.dtype, numel: int) -> None:
    impls = testing.get_impls(
        python_impl=python_scal.scal,
        pytorch_impl=pytorch_scal.scal if HAS_PYTORCH else None,
        triton_impl=triton_scal.scal if HAS_TRITON else None,
        cute_impl=cute_scal.scal if HAS_CUTE else None,
    )

    # Benchmark each implementation
    config = testing.BenchmarkConfig(warmup=3, repeat=1_000)
    results = testing.run_benchmarks(
        impls,
        lambda: factory(numel, device, dtype),
        flops=2.0 * numel,
        config=config,
    )

    testing.show_benchmarks(results)
