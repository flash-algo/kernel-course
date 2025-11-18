from typing import List

import pytest
import torch

from kernel_course import testing
from kernel_course.python_ops import copy as python_copy

try:
	from kernel_course.pytorch_ops import copy as pytorch_copy
	HAS_PYTORCH = True
except Exception:
	pytorch_copy = None
	HAS_PYTORCH = False

try:
	from kernel_course.triton_ops import copy as triton_copy
	HAS_TRITON = True
except Exception:
	triton_copy = None
	HAS_TRITON = False

try:
	from kernel_course.cute_ops import copy as cute_copy
	HAS_CUTE = True
except Exception:
	cute_copy = None
	HAS_CUTE = False


def factory(
    numel: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
):
    x = torch.linspace(0.0, 1.0, steps=numel, device=device, dtype=dtype)
    y = torch.empty_like(x)
    return (x, y), {}


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
    [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.bool],
)
def test_copy_benchmark(device: torch.device, dtype: torch.dtype) -> None:
	impls: List[testing.Implementation] = [
		testing.Implementation("python", python_copy.copy, testing.Backend.PYTHON),
	]
	impls.append(
		testing.Implementation("pytorch", pytorch_copy.copy, testing.Backend.PYTORCH)
	) if HAS_PYTORCH else None
	impls.append(
		testing.Implementation("triton", triton_copy.copy, testing.Backend.TRITON)
	) if HAS_TRITON else None
	impls.append(
		testing.Implementation("cute", cute_copy.copy, testing.Backend.CUTE)
	) if HAS_CUTE else None

	numel = 1 << 16
	# Benchmark each implementation
	config = testing.BenchmarkConfig(warmup=3, repeat=1_000)
	results = testing.run_benchmarks(
		impls,
		lambda: factory(numel, device, dtype),
		flops=0.0,
		config=config,
	)

	testing.show_benchmarks(results)
