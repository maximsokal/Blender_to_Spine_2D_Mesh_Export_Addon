"""Mutable runtime counters used only inside one pipeline trace session."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


class FunctionTraceStats:
    __slots__ = (
        "module", "relative_path", "function", "first_line", "call_count",
        "return_count", "exception_event_count", "inclusive_ns", "self_ns",
        "max_ns", "input_signatures", "output_signatures", "exception_types",
    )

    def __init__(self, *, module: str, relative_path: str, function: str, first_line: int) -> None:
        self.module = module
        self.relative_path = relative_path
        self.function = function
        self.first_line = first_line
        self.call_count = 0
        self.return_count = 0
        self.exception_event_count = 0
        self.inclusive_ns = 0
        self.self_ns = 0
        self.max_ns = 0
        self.input_signatures: dict[str, Mapping[str, Any]] = {}
        self.output_signatures: dict[str, Mapping[str, Any]] = {}
        self.exception_types: Counter[str] = Counter()


class ActiveTraceCall:
    __slots__ = ("frame_id", "key", "started_ns", "child_ns")

    def __init__(self, frame_id: int, key: tuple[str, str, int], started_ns: int) -> None:
        self.frame_id = frame_id
        self.key = key
        self.started_ns = started_ns
        self.child_ns = 0


__all__ = ["ActiveTraceCall", "FunctionTraceStats"]
