"""Lightweight benchmarking helpers for course exercises.

This module provides a small harness that test cases can use to compare
multiple kernel implementations (plain Python, PyTorch, Triton, CuTe, ...)
for both correctness and wall-clock performance.  A typical workflow is:

.. code-block:: python

    from kernel_course import testing
    from kernel_course.python_ops import add as python_impl
    from kernel_course.triton_ops import add as triton_impl

    inputs = testing.static_input_factory(a, b)
    implementations = [
        testing.Implementation("python", python_impl.add, testing.Backend.PYTHON),
        testing.Implementation("torch", torch.add, testing.Backend.PYTORCH),
        testing.Implementation("triton", triton_impl.add, testing.Backend.TRITON),
    ]

    results = testing.benchmark_many(implementations, inputs)
    testing.assert_close_to_baseline(results)
    print(testing.format_results(results))

The harness keeps the API surface intentionally small so that individual
exercises can extend it with their own convenience wrappers if desired.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import mean
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

__all__ = [
    "Backend",
    "BenchmarkConfig",
    "BenchmarkResult",
    "Implementation",
    "static_input_factory",
    "benchmark_impl",
    "benchmark_many",
    "assert_close_to_baseline",
    "format_results",
]


class Backend(str, Enum):
    """Identifier for the execution environment of an implementation."""

    PYTHON = "python"
    PYTORCH = "pytorch"
    TRITON = "triton"
    CUTE = "cute"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


def _torch_synchronize_if_available() -> None:
    """Synchronize the CUDA device when available."""

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

    def speedup_vs(self, baseline: "BenchmarkResult") -> float:
        return baseline.mean_ms / self.mean_ms


ArgsFactory = Callable[[], Tuple[Tuple[Any, ...], Dict[str, Any]]]


def _clone_tree(value: Any) -> Any:
    """Clone torch tensors inside arbitrarily nested containers."""

    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, (list, tuple)):
        cloned = [_clone_tree(v) for v in value]
        return type(value)(cloned)
    if isinstance(value, dict):
        return {k: _clone_tree(v) for k, v in value.items()}
    if isinstance(value, set):
        return type(value)(_clone_tree(v) for v in value)
    return value


def static_input_factory(*args: Any, **kwargs: Any) -> ArgsFactory:
    """Return a callable that recreates the provided inputs for each run."""

    def factory() -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        return _clone_tree(args), _clone_tree(kwargs)

    return factory


def _run_once(impl: Implementation, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    impl.synchronize()
    result = impl(*args, **kwargs)
    impl.synchronize()
    return result


def benchmark_impl(
    impl: Implementation,
    input_factory: ArgsFactory,
    config: BenchmarkConfig | None = None,
) -> BenchmarkResult:
    """Benchmark a single implementation using the provided input factory."""

    cfg = config or BenchmarkConfig()

    # Warmup runs to amortize setup and cache effects.
    for _ in range(cfg.warmup):
        args, kwargs = input_factory()
        _run_once(impl, args, kwargs)

    timings_ms: List[float] = []
    final_output: Any = None
    for _ in range(cfg.repeat):
        args, kwargs = input_factory()
        impl.synchronize()
        start = perf_counter()
        final_output = impl(*args, **kwargs)
        impl.synchronize()
        timings_ms.append((perf_counter() - start) * 1e3)

    assert final_output is not None
    return BenchmarkResult(impl=impl, timings_ms=timings_ms, output=final_output)


def benchmark_many(
    implementations: Sequence[Implementation],
    input_factory: ArgsFactory,
    config: BenchmarkConfig | None = None,
) -> List[BenchmarkResult]:
    """Benchmark a collection of implementations with shared inputs."""

    if not implementations:
        raise ValueError("No implementations provided for benchmarking")

    cfg = config or BenchmarkConfig()
    results = [benchmark_impl(impl, input_factory, cfg) for impl in implementations]
    return results


def _assert_close(a: Any, b: Any, *, rtol: float, atol: float) -> None:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            raise AssertionError("Sequences do not share the same length")
        for lhs, rhs in zip(a, b):
            _assert_close(lhs, rhs, rtol=rtol, atol=atol)
        return
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise AssertionError("Dictionaries do not share the same keys")
        for key in a:
            _assert_close(a[key], b[key], rtol=rtol, atol=atol)
        return
    if a != b:
        raise AssertionError(f"Values differ: {a!r} != {b!r}")


def assert_close_to_baseline(
    results: Sequence[BenchmarkResult],
    *,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> None:
    """Check every result against the first one, used as baseline."""

    if not results:
        raise ValueError("No results to validate")

    baseline = results[0]
    for candidate in results[1:]:
        try:
            _assert_close(candidate.output, baseline.output, rtol=rtol, atol=atol)
        except AssertionError as exc:  # pragma: no cover - tiny wrapper
            raise AssertionError(
                f"Implementation '{candidate.impl.name}' diverged from baseline "
                f"'{baseline.impl.name}'"
            ) from exc


def format_results(results: Sequence[BenchmarkResult]) -> str:
    """Render benchmark results as a small text table."""

    if not results:
        return "<no results>"

    baseline = results[0]
    rows = ["name backend mean_ms best_ms speedup"]
    for entry in results:
        speedup = entry.speedup_vs(baseline) if entry is not baseline else 1.0
        rows.append(
            f"{entry.impl.name} {entry.impl.backend.value} "
            f"{entry.mean_ms:8.3f} {entry.best_ms:8.3f} {speedup:7.2f}"
        )
    return "\n".join(rows)
