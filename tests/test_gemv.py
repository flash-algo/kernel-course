import pytest
import torch

from kernel_course import testing
from kernel_course.python_ops import gemv as python_gemv

try:
    from kernel_course.pytorch_ops import gemv as pytorch_gemv

    HAS_PYTORCH = True
except Exception:
    pytorch_gemv = None
    HAS_PYTORCH = False

try:
    from kernel_course.triton_ops import gemv as triton_gemv

    HAS_TRITON = True
except Exception:
    triton_gemv = None
    HAS_TRITON = False

try:
    from kernel_course.cute_ops import gemv as cute_gemv

    HAS_CUTE = True
except Exception:
    cute_gemv = None
    HAS_CUTE = False


def factory(
    MN: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    M, N = MN
    A = torch.linspace(0.0, 1.0, steps=M * N, device=device, dtype=dtype).view(M, N)
    x = torch.linspace(0.0, 1.0, steps=N, device=device, dtype=dtype)
    y = torch.linspace(0.0, 1.0, steps=M, device=device, dtype=dtype)
    alpha = 1.14
    beta = 5.14
    return (A, x, y, alpha, beta), {}


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
    "MN",
    [
        (1 << 4, 1 << 4),
        (1 << 8, 1 << 8),
    ],
)
def test_gemv(
    device: torch.device,
    dtype: torch.dtype,
    MN: tuple[int, int],
) -> None:
    impls = testing.get_impls(
        python_impl=python_gemv.gemv,
        pytorch_impl=pytorch_gemv.gemv if HAS_PYTORCH else None,
        triton_impl=triton_gemv.gemv if HAS_TRITON else None,
        cute_impl=cute_gemv.gemv if HAS_CUTE else None,
    )

    # Benchmark each implementation
    config = testing.BenchmarkConfig(warmup=3, repeat=100)
    results = testing.run_benchmarks(
        impls,
        lambda: factory(MN, device, dtype),
        flops=2 * MN[0] * MN[1],
        config=config,
    )

    testing.show_benchmarks(results)
