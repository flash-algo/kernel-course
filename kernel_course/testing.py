from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "Backend",
    "Implementation",
    "BenchmarkConfig",
    "BenchmarkResult",
    "run_benchmarks",
    "show_benchmarks",
]


class Backend(str, Enum):

    PYTHON = "python"
    PYTORCH = "pytorch"
    TRITON = "triton"
    CUTE = "cute"

    def __str__(self) -> str:
        return self.value


def _torch_synchronize_if_available() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


_SYNCHRONIZERS = {
    Backend.PYTHON: lambda: None,
    Backend.PYTORCH: _torch_synchronize_if_available,
    Backend.TRITON: _torch_synchronize_if_available,
    Backend.CUTE: _torch_synchronize_if_available,
}


@dataclass
class Implementation:
    """Wrapper describing a concrete kernel implementation."""

    name: str
    fn: Callable[..., Any]
    backend: Backend
    description: Optional[str] = None
    synchronizer: Optional[Callable[[], None]] = None

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise TypeError(f"Implementation '{self.name}' must wrap a callable.")
        if self.synchronizer is None:
            self.synchronizer = _SYNCHRONIZERS.get(self.backend, lambda: None)

    def synchronize(self) -> None:
        assert self.synchronizer is not None
        self.synchronizer()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    warmup: int = 5
    repeat: int = 20

    def __post_init__(self) -> None:
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.repeat <= 0:
            raise ValueError("repeat must be greater than zero")


@dataclass
class BenchmarkResult:
    """Outcome of benchmarking a single implementation."""

    impl: Implementation
    timings_ms: Sequence[float]
    flops: float
    output: Any

    @property
    def mean_ms(self) -> float:
        return mean(self.timings_ms)

    @property
    def best_ms(self) -> float:
        return min(self.timings_ms)

    @property
    def worst_ms(self) -> float:
        return max(self.timings_ms)

    @property
    def tflops(self) -> float:
        seconds = self.mean_ms / 1e3
        return self.flops / seconds / 1e12

    def speedup_vs(self, baseline: "BenchmarkResult") -> float:
        return baseline.mean_ms / self.mean_ms


def _run_benchmark(
    impl: Implementation,
    factory: Callable[[], Tuple[Tuple[Any, ...], Dict[str, Any]]],
    *,
    flops: float,
    config: Optional[BenchmarkConfig] = None,
) -> BenchmarkResult:
    """
    Run benchmark for a single implementation.

    Args:
        impl: The implementation to benchmark.
        factory: Callable producing ``(args, kwargs)`` per invocation. It should
            return a tuple where the first element is a tuple of positional
            arguments and the second element is a dict of keyword arguments.
        flops: Total floating-point operations performed by one invocation.
        config: Benchmark configuration. Defaults to ``BenchmarkConfig()``.

    Returns:
        BenchmarkResult: Aggregated timings, last output, and FLOPs metadata.
    """
    if config is None:
        config = BenchmarkConfig()

    # Warmup (not timed)
    for _ in range(config.warmup):
        args, kwargs = factory()
        impl.synchronize()
        _ = impl(*args, **kwargs)
        impl.synchronize()

    # Timed repeats
    timings_ms: List[float] = []
    output: Any = None
    for _ in range(config.repeat):
        args, kwargs = factory()
        impl.synchronize()
        start = perf_counter()
        output = impl(*args, **kwargs)
        impl.synchronize()
        end = perf_counter()
        timings_ms.append((end - start) * 1e3)

    return BenchmarkResult(
        impl=impl,
        timings_ms=timings_ms,
        flops=flops,
        output=output,
    )


def run_benchmarks(
    impls: Iterable[Implementation],
    factory: Callable[[], Tuple[Tuple[Any, ...], Dict[str, Any]]],
    *,
    flops: float,
    config: Optional[BenchmarkConfig] = None,
) -> List[BenchmarkResult]:
    """
    Run benchmarks for multiple implementations, optionally validating outputs.

    The first implementation is treated as the numerical baseline.
    If `validate` is True, every other implementation's single sample output 
    is compared against the baseline output produced from its own fresh
    factory invocation.
    """
    impl_list = list(impls)
    if not impl_list:
        return []

    # Establish baseline output
    baseline_impl = impl_list[0]
    base_args, base_kwargs = factory()
    baseline_output = baseline_impl(*base_args, **base_kwargs)
    if baseline_output.dtype == torch.float32:
        rtol = 1e-5
        atol = 1e-8
    elif baseline_output.dtype == torch.float16:
        rtol = 1e-3
        atol = 1e-5
    elif baseline_output.dtype == torch.bfloat16:
        rtol = 1e-2
        atol = 1e-3
    else:
        rtol = 1e-5
        atol = 1e-8

    results: List[BenchmarkResult] = []
    for impl in impl_list:
        args, kwargs = factory()
        out = impl(*args, **kwargs)

        if not torch.allclose(out, baseline_output, rtol=rtol, atol=atol):
            raise AssertionError(
                f"Output mismatch vs baseline for '{impl.name}' backend={impl.backend}"
            )

        res = _run_benchmark(impl, factory, flops=flops, config=config)
        results.append(res)
    return results


def show_benchmarks(results: Sequence[BenchmarkResult]) -> None:
    """
    Pretty-print benchmark results.

    If multiple results are provided, the first is treated as the baseline for
    speedup computation.
    """
    if not results:
        print("No results to display.")
        return

    baseline = results[0]

    # Header
    headers = (
        "backend",
        "speed (ms)",
        "speedup",
        "tflops",
    )
    print(
        f"\n{headers[0]:<10} {headers[1]:>10} {headers[2]:>8} {headers[3]:>10}"
    )
    print("-" * 42)

    for r in results:
        speed = r.speedup_vs(baseline)
        tflops = r.tflops if r.flops > 0 else 0.0
        print(
            f"{str(r.impl.backend):<10} "
            f"{r.mean_ms:>10.3f} "
            f"{speed:>8.2f} {tflops:>10.3f}"
        )

