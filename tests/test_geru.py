import pytest
import torch

from kernel_course import testing
from kernel_course.python_ops import geru as python_geru

try:
    from kernel_course.pytorch_ops import geru as pytorch_geru

    HAS_PYTORCH = True
except Exception:
    pytorch_geru = None
    HAS_PYTORCH = False

try:
    from kernel_course.triton_ops import geru as triton_geru

    HAS_TRITON = True
except Exception:
    triton_geru = None
    HAS_TRITON = False

try:
    from kernel_course.cute_ops import geru as cute_geru

    HAS_CUTE = True
except Exception:
    cute_geru = None
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
    alpha = 3.14
    return (A, x, y, alpha), {}


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
    [
        (1 << 4, 1 << 4),
        (1 << 8, 1 << 8),
    ],
)
def test_geru_benchmark(
    device: torch.device, dtype: torch.dtype, numel: tuple[int, int]
) -> None:
    impls = testing.get_impls(
        python_impl=python_geru.geru,
        pytorch_impl=pytorch_geru.geru if HAS_PYTORCH else None,
        triton_impl=triton_geru.geru if HAS_TRITON else None,
        cute_impl=cute_geru.geru if HAS_CUTE else None,
    )

    # Benchmark each implementation
    config = testing.BenchmarkConfig(warmup=3, repeat=1_000)
    results = testing.run_benchmarks(
        impls,
        lambda: factory(numel, device, dtype),
        flops=2 * numel[0] * numel[1],
        config=config,
    )

    testing.show_benchmarks(results)
