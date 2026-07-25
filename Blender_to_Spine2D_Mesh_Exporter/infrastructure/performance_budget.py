"""Relative, machine-independent performance regression budgets."""

from __future__ import annotations
from dataclasses import dataclass
import math
from statistics import median
import time
from typing import Callable, Iterable

@dataclass(frozen=True, slots=True)
class PerformanceSample:
    size: int
    seconds: float
    def __post_init__(self) -> None:
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size <= 0: raise ValueError("size must be a positive integer")
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float)) or not math.isfinite(float(self.seconds)) or self.seconds <= 0.0: raise ValueError("seconds must be finite and positive")

@dataclass(frozen=True, slots=True)
class RelativePerformanceBudget:
    maximum_time_ratio_per_size_ratio: float = 3.0
    minimum_measured_seconds: float = 1e-6
    def __post_init__(self) -> None:
        for field_name in ("maximum_time_ratio_per_size_ratio", "minimum_measured_seconds"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0.0: raise ValueError(f"{field_name} must be finite and positive")
    def assert_within(self, samples: Iterable[PerformanceSample]) -> None:
        values = tuple(sorted(samples, key=lambda item: item.size))
        if len(values) < 2: raise ValueError("at least two performance samples are required")
        for previous, current in zip(values, values[1:], strict=False):
            size_ratio = current.size / previous.size
            time_ratio = max(current.seconds, self.minimum_measured_seconds) / max(previous.seconds, self.minimum_measured_seconds)
            normalized = time_ratio / size_ratio
            if normalized > self.maximum_time_ratio_per_size_ratio:
                raise AssertionError(f"performance growth exceeded budget: size {previous.size}->{current.size} ({size_ratio:.3f}x), time {previous.seconds:.6f}->{current.seconds:.6f}s ({time_ratio:.3f}x), normalized={normalized:.3f}, budget={self.maximum_time_ratio_per_size_ratio:.3f}")

def measure_median(function: Callable[[], object], *, repeats: int = 5, warmups: int = 1) -> float:
    if not callable(function): raise TypeError("function must be callable")
    for name, value, minimum in (("repeats", repeats, 1), ("warmups", warmups, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum: raise ValueError(f"{name} must be an integer >= {minimum}")
    for _ in range(warmups): function()
    values = []
    for _ in range(repeats):
        started = time.perf_counter(); function(); values.append(max(time.perf_counter() - started, 1e-12))
    return float(median(values))

__all__ = ["PerformanceSample", "RelativePerformanceBudget", "measure_median"]
