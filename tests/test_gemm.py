import pytest
import torch

from kernel_course import testing
from kernel_course.python_ops import gemm as python_gemm

try:
    from kernel_course.pytorch_ops import gemm as pytorch_gemm

    HAS_PYTORCH = True
except Exception:
    pytorch_gemm = None
    HAS_PYTORCH = False

try:
    from kernel_course.triton_ops import gemm as triton_gemm

    HAS_TRITON = True
except Exception:
    triton_gemm = None
    HAS_TRITON = False

try:
    from kernel_course.cute_ops import gemm as cute_gemm

    HAS_CUTE = True
except Exception:
    cute_gemm = None
    HAS_CUTE = False


def factory(
    MNK: tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    M, N, K = MNK
    A = torch.linspace(0.0, 1.0, steps=M * K, device=device, dtype=dtype).view(M, K)
    B = torch.linspace(0.0, 1.0, steps=K * N, device=device, dtype=dtype).view(K, N)
    C = torch.linspace(0.0, 1.0, steps=M * N, device=device, dtype=dtype).view(M, N)
    alpha = 1.14
    beta = 5.14
    return (A, B, C, alpha, beta), {}


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
    "MNK",
    [
        (1 << 4, 1 << 4, 1 << 4),
        (1 << 8, 1 << 8, 1 << 8),
    ],
)
def test_gemm_benchmark(
    device: torch.device,
    dtype: torch.dtype,
    MNK: tuple[int, int, int],
) -> None:
    impls = testing.get_impls(
        python_impl=python_gemm.gemm,
        pytorch_impl=pytorch_gemm.gemm if HAS_PYTORCH else None,
        triton_impl=triton_gemm.gemm if HAS_TRITON else None,
        cute_impl=cute_gemm.gemm if HAS_CUTE else None,
    )

    # Benchmark each implementation
    config = testing.BenchmarkConfig(warmup=3, repeat=100)
    results = testing.run_benchmarks(
        impls,
        lambda: factory(MNK, device, dtype),
        flops=2 * MNK[0] * MNK[1] * MNK[2],
        config=config,
    )

    testing.show_benchmarks(results)
