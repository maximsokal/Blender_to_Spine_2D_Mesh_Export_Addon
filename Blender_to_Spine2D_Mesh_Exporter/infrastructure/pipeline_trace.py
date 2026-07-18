"""Low-overhead runtime tracing for one Blender-to-Spine export scenario."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys
import threading
from time import perf_counter_ns
from types import FrameType
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

from .pipeline_trace_model import ActiveTraceCall, FunctionTraceStats
from .pipeline_trace_report import build_trace_report
from .pipeline_trace_values import (
    argument_names,
    canonical_json,
    discover_pipeline_modules,
    shape_summary,
)


DEFAULT_MAX_EVENTS = 250_000
DEFAULT_SIGNATURE_SAMPLES = 4


class PipelineTraceError(RuntimeError):
    """Raised when a trace session is configured or consumed incorrectly."""


class PipelineTraceSession:
    """Observe all addon Python calls during one explicitly bounded operation."""

    def __init__(
        self,
        package_directory: Path,
        *,
        package_name: str,
        focus_modules: Sequence[str] = (),
        max_events: int = DEFAULT_MAX_EVENTS,
        capture_values: bool = False,
        signature_samples: int = DEFAULT_SIGNATURE_SAMPLES,
    ) -> None:
        if not isinstance(package_directory, Path):
            raise TypeError("package_directory must be pathlib.Path")
        if not isinstance(package_name, str) or not package_name.strip():
            raise ValueError("package_name must be a non-empty string")
        if not isinstance(max_events, int) or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        if not isinstance(signature_samples, int) or signature_samples < 1:
            raise ValueError("signature_samples must be a positive integer")

        self.package_directory = package_directory.expanduser().resolve(strict=False)
        self.package_name = package_name.strip()
        self.focus_modules = tuple(
            value.strip() for value in focus_modules
            if isinstance(value, str) and value.strip()
        )
        self.max_events = max_events
        self.capture_values = bool(capture_values)
        self.signature_samples = signature_samples
        self.run_id = uuid4().hex
        self._modules_by_file = discover_pipeline_modules(self.package_directory)
        self._stats: dict[tuple[str, str, int], FunctionTraceStats] = {}
        self._edges: Counter[tuple[tuple[str, str, int], tuple[str, str, int]]] = Counter()
        self._stacks: dict[int, list[ActiveTraceCall]] = defaultdict(list)
        self._event_count = 0
        self._truncated = False
        self._focus_timeline: list[dict[str, Any]] = []
        self._focus_timeline_limit = min(10_000, max_events)
        self._focus_timeline_truncated = False
        self._started_ns: int | None = None
        self._stopped_ns: int | None = None
        self._started_at: str | None = None
        self._previous_trace: Any = None
        self._previous_thread_trace: Any = None
        self._running = False

    def __enter__(self) -> "PipelineTraceSession":
        self.start()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.stop()

    def start(self) -> None:
        if self._running:
            raise PipelineTraceError("trace session is already running")
        self._started_ns = perf_counter_ns()
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._previous_trace = sys.gettrace()
        get_thread_trace = getattr(threading, "gettrace", None)
        self._previous_thread_trace = get_thread_trace() if get_thread_trace else None
        self._running = True
        sys.settrace(self._trace)
        threading.settrace(self._trace)

    def stop(self) -> None:
        if not self._running:
            return
        sys.settrace(self._previous_trace)
        threading.settrace(self._previous_thread_trace)
        self._stopped_ns = perf_counter_ns()
        self._running = False

    def _file_metadata(self, frame: FrameType) -> dict[str, str] | None:
        try:
            path = Path(frame.f_code.co_filename).resolve(strict=False)
        except (TypeError, ValueError, OSError):
            return None
        metadata = self._modules_by_file.get(path)
        if metadata is not None and metadata["module"].startswith("infrastructure.pipeline_trace"):
            return None
        return metadata

    @staticmethod
    def _function_key(frame: FrameType, metadata: Mapping[str, str]) -> tuple[str, str, int]:
        qualname = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        return metadata["module"], str(qualname), int(frame.f_code.co_firstlineno)

    def _record_signature(
        self,
        target: dict[str, Mapping[str, Any]],
        signature: Mapping[str, Any],
    ) -> None:
        if len(target) >= self.signature_samples:
            return
        target.setdefault(canonical_json(signature), signature)

    def _focus_matches(self, module: str, relative_path: str) -> bool:
        if not self.focus_modules:
            return False
        return any(
            query.casefold() in haystack
            for query in self.focus_modules
            for haystack in (module.casefold(), relative_path.casefold())
        )

    def _append_focus_event(self, event: Mapping[str, Any]) -> None:
        if len(self._focus_timeline) < self._focus_timeline_limit:
            self._focus_timeline.append(dict(event))
        else:
            self._focus_timeline_truncated = True

    def _trace(self, frame: FrameType, event: str, arg: Any):
        metadata = self._file_metadata(frame)
        if metadata is None:
            return None

        frame.f_trace_lines = False
        frame.f_trace_opcodes = False
        thread_id = threading.get_ident()
        stack = self._stacks[thread_id]

        if event == "call":
            if self._event_count >= self.max_events:
                self._truncated = True
                return self._trace
            self._event_count += 1
            key = self._function_key(frame, metadata)
            stats = self._stats.get(key)
            if stats is None:
                stats = FunctionTraceStats(
                    module=metadata["module"],
                    relative_path=metadata["relative_path"],
                    function=key[1],
                    first_line=key[2],
                )
                self._stats[key] = stats
            stats.call_count += 1
            if stack:
                self._edges[(stack[-1].key, key)] += 1
            stack.append(ActiveTraceCall(id(frame), key, perf_counter_ns()))

            signature = {
                name: shape_summary(frame.f_locals[name], capture_values=self.capture_values)
                for name in argument_names(frame) if name in frame.f_locals
            }
            self._record_signature(stats.input_signatures, signature)
            if self._focus_matches(metadata["module"], metadata["relative_path"]):
                self._append_focus_event({
                    "sequence": self._event_count,
                    "event": "call",
                    "thread_id": thread_id,
                    "depth": len(stack) - 1,
                    "module": metadata["module"],
                    "relative_path": metadata["relative_path"],
                    "function": key[1],
                    "first_line": key[2],
                    "input": signature,
                })
            return self._trace

        if not stack:
            return self._trace
        active = stack[-1]
        if active.frame_id != id(frame):
            return self._trace
        stats = self._stats[active.key]

        if event == "exception":
            self._event_count += 1
            stats.exception_event_count += 1
            try:
                exception_type = arg[0]
                name = getattr(exception_type, "__name__", str(exception_type))
            except Exception:  # pipeline-audit: ignore=BROAD_EXCEPTION_WITHOUT_VISIBILITY
                name = "<unknown>"
            stats.exception_types[str(name)] += 1
            if self._focus_matches(metadata["module"], metadata["relative_path"]):
                self._append_focus_event({
                    "sequence": self._event_count,
                    "event": "exception",
                    "thread_id": thread_id,
                    "depth": len(stack) - 1,
                    "module": metadata["module"],
                    "relative_path": metadata["relative_path"],
                    "function": active.key[1],
                    "first_line": active.key[2],
                    "exception_type": str(name),
                })
            return self._trace

        if event == "return":
            self._event_count += 1
            stack.pop()
            elapsed = max(0, perf_counter_ns() - active.started_ns)
            stats.return_count += 1
            stats.inclusive_ns += elapsed
            stats.self_ns += max(0, elapsed - active.child_ns)
            stats.max_ns = max(stats.max_ns, elapsed)
            output = shape_summary(arg, capture_values=self.capture_values)
            self._record_signature(stats.output_signatures, output)
            if self._focus_matches(metadata["module"], metadata["relative_path"]):
                self._append_focus_event({
                    "sequence": self._event_count,
                    "event": "return",
                    "thread_id": thread_id,
                    "depth": len(stack),
                    "module": metadata["module"],
                    "relative_path": metadata["relative_path"],
                    "function": active.key[1],
                    "first_line": active.key[2],
                    "duration_ms": round(elapsed / 1_000_000.0, 6),
                    "output": output,
                })
            if stack:
                stack[-1].child_ns += elapsed
            return self._trace
        return self._trace

    def build_report(
        self,
        *,
        run_success: bool,
        run_error: Mapping[str, Any] | None = None,
        scenario: str | None = None,
        expected_calls: Iterable[tuple[str, str]] = (),
    ) -> dict[str, Any]:
        """Build one deterministic JSON-compatible report after the session stops."""

        if self._running:
            self.stop()
        if self._started_ns is None:
            raise PipelineTraceError("trace session was never started")
        return build_trace_report(
            run_id=self.run_id,
            scenario=scenario,
            package_name=self.package_name,
            package_directory=str(self.package_directory),
            started_at_utc=self._started_at,
            started_ns=self._started_ns,
            stopped_ns=self._stopped_ns,
            run_success=run_success,
            run_error=run_error,
            event_count=self._event_count,
            event_limit=self.max_events,
            truncated=self._truncated,
            capture_values=self.capture_values,
            modules_by_file=self._modules_by_file,
            stats_by_key=self._stats,
            edges=self._edges,
            imported_modules=sys.modules,
            focus_queries=self.focus_modules,
            focus_timeline=self._focus_timeline,
            focus_timeline_truncated=self._focus_timeline_truncated,
            expected_calls=expected_calls,
        )


__all__ = [
    "PipelineTraceError",
    "PipelineTraceSession",
    "discover_pipeline_modules",
]
