"""Deterministic JSON report assembly for runtime pipeline traces."""

from __future__ import annotations

from collections import defaultdict
from time import perf_counter_ns
from typing import Any, Iterable, Mapping, Sequence

from .pipeline_trace_model import FunctionTraceStats
from .pipeline_trace_values import PRODUCTION_LAYERS


SCHEMA_VERSION = 1
DEFAULT_HOTSPOT_COUNT = 25


def function_payload(stats: FunctionTraceStats) -> dict[str, Any]:
    return {
        "function": stats.function,
        "first_line": stats.first_line,
        "call_count": stats.call_count,
        "return_count": stats.return_count,
        "exception_event_count": stats.exception_event_count,
        "inclusive_ms": round(stats.inclusive_ns / 1_000_000.0, 6),
        "self_ms": round(stats.self_ns / 1_000_000.0, 6),
        "max_call_ms": round(stats.max_ns / 1_000_000.0, 6),
        "exception_types": dict(stats.exception_types),
        "input_signatures": list(stats.input_signatures.values()),
        "output_signatures": list(stats.output_signatures.values()),
    }


def module_records(
    *,
    modules_by_file: Mapping[Any, Mapping[str, str]],
    stats_by_key: Mapping[tuple[str, str, int], FunctionTraceStats],
    package_name: str,
    imported_modules: Mapping[str, Any],
) -> list[dict[str, Any]]:
    functions_by_module: dict[str, list[FunctionTraceStats]] = defaultdict(list)
    for stats in stats_by_key.values():
        functions_by_module[stats.module].append(stats)

    records: list[dict[str, Any]] = []
    for _path, metadata in sorted(
        modules_by_file.items(),
        key=lambda item: item[1]["relative_path"].casefold(),
    ):
        module = metadata["module"]
        function_stats = sorted(
            functions_by_module.get(module, ()),
            key=lambda value: (value.first_line, value.function),
        )
        imported_name = package_name
        if module not in {"<package>", ""}:
            imported_name = f"{package_name}.{module}"
        status = (
            "executed" if function_stats else
            "imported_not_called" if imported_name in imported_modules else
            "not_imported"
        )
        records.append({
            "module": module,
            "relative_path": metadata["relative_path"],
            "layer": metadata["layer"],
            "status": status,
            "call_count": sum(item.call_count for item in function_stats),
            "exception_event_count": sum(item.exception_event_count for item in function_stats),
            "inclusive_ms": round(sum(item.inclusive_ns for item in function_stats) / 1_000_000.0, 6),
            "self_ms": round(sum(item.self_ns for item in function_stats) / 1_000_000.0, 6),
            "functions": [function_payload(item) for item in function_stats],
        })
    return records


def edge_records(edges: Mapping[Any, int]) -> list[dict[str, Any]]:
    ordered = sorted(edges.items(), key=lambda item: item[1], reverse=True)
    return [{
        "source": {"module": source[0], "function": source[1], "first_line": source[2]},
        "target": {"module": target[0], "function": target[1], "first_line": target[2]},
        "call_count": count,
    } for (source, target), count in ordered]


def hotspot_records(stats_by_key: Mapping[Any, FunctionTraceStats]) -> list[dict[str, Any]]:
    ordered = sorted(
        stats_by_key.values(),
        key=lambda value: (value.self_ns, value.inclusive_ns),
        reverse=True,
    )[:DEFAULT_HOTSPOT_COUNT]
    return [{
        "module": stats.module,
        "relative_path": stats.relative_path,
        **function_payload(stats),
    } for stats in ordered]


def missing_expected_calls(
    stats_by_key: Mapping[Any, FunctionTraceStats],
    expected_calls: Iterable[tuple[str, str]],
) -> list[dict[str, str]]:
    observed = {(stats.module, stats.function) for stats in stats_by_key.values()}
    return [
        {"module": module, "function_contains": function_fragment}
        for module, function_fragment in expected_calls
        if not any(
            observed_module == module and function_fragment in observed_function
            for observed_module, observed_function in observed
        )
    ]


def exception_hotspots(stats_by_key: Mapping[Any, FunctionTraceStats]) -> list[dict[str, Any]]:
    ordered = sorted(
        (item for item in stats_by_key.values() if item.exception_event_count),
        key=lambda value: value.exception_event_count,
        reverse=True,
    )
    return [{
        "module": stats.module,
        "relative_path": stats.relative_path,
        "function": stats.function,
        "first_line": stats.first_line,
        "exception_event_count": stats.exception_event_count,
        "exception_types": dict(stats.exception_types),
    } for stats in ordered]


def focus_payload(
    *,
    focus_queries: Sequence[str],
    module_records_value: Sequence[Mapping[str, Any]],
    edge_records_value: Sequence[Mapping[str, Any]],
    timeline: Sequence[Mapping[str, Any]],
    timeline_truncated: bool,
) -> dict[str, Any]:
    def matches(module: str, relative_path: str) -> bool:
        return any(
            query.casefold() in haystack
            for query in focus_queries
            for haystack in (module.casefold(), relative_path.casefold())
        )

    matched_modules = [
        dict(record) for record in module_records_value
        if matches(str(record["module"]), str(record["relative_path"]))
    ]
    names = {str(record["module"]) for record in matched_modules}
    edges = [
        dict(edge) for edge in edge_records_value
        if str(edge["source"]["module"]) in names
        or str(edge["target"]["module"]) in names
    ]
    return {
        "queries": list(focus_queries),
        "matched_modules": matched_modules,
        "call_edges": edges,
        "timeline": list(timeline),
        "timeline_truncated": timeline_truncated,
    }


def build_trace_report(
    *,
    run_id: str,
    scenario: str | None,
    package_name: str,
    package_directory: str,
    started_at_utc: str | None,
    started_ns: int,
    stopped_ns: int | None,
    run_success: bool,
    run_error: Mapping[str, Any] | None,
    event_count: int,
    event_limit: int,
    truncated: bool,
    capture_values: bool,
    modules_by_file: Mapping[Any, Mapping[str, str]],
    stats_by_key: Mapping[tuple[str, str, int], FunctionTraceStats],
    edges: Mapping[Any, int],
    imported_modules: Mapping[str, Any],
    focus_queries: Sequence[str],
    focus_timeline: Sequence[Mapping[str, Any]],
    focus_timeline_truncated: bool,
    expected_calls: Iterable[tuple[str, str]],
) -> dict[str, Any]:
    stopped = stopped_ns or perf_counter_ns()
    modules = module_records(
        modules_by_file=modules_by_file,
        stats_by_key=stats_by_key,
        package_name=package_name,
        imported_modules=imported_modules,
    )
    calls = edge_records(edges)
    missing = missing_expected_calls(stats_by_key, expected_calls)
    production = [record for record in modules if record["layer"] in PRODUCTION_LAYERS]
    executed = [record for record in modules if record["status"] == "executed"]
    executed_production = [record for record in production if record["status"] == "executed"]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "scenario": scenario,
        "package_name": package_name,
        "package_directory": package_directory,
        "started_at_utc": started_at_utc,
        "duration_ms": round((stopped - started_ns) / 1_000_000.0, 6),
        "run_success": bool(run_success),
        "run_error": None if run_error is None else dict(run_error),
        "event_count": event_count,
        "event_limit": event_limit,
        "truncated": truncated,
        "capture_values": capture_values,
        "summary": {
            "discovered_module_count": len(modules),
            "production_module_count": len(production),
            "executed_module_count": len(executed),
            "executed_production_module_count": len(executed_production),
            "production_coverage_ratio": 0.0 if not production else round(len(executed_production) / len(production), 6),
            "function_count": len(stats_by_key),
            "exception_event_count": sum(item.exception_event_count for item in stats_by_key.values()),
            "missing_expected_call_count": len(missing),
        },
        "missing_expected_calls": missing,
        "modules": modules,
        "call_edges": calls,
        "hotspots": hotspot_records(stats_by_key),
        "exception_hotspots": exception_hotspots(stats_by_key),
        "focus": focus_payload(
            focus_queries=focus_queries,
            module_records_value=modules,
            edge_records_value=calls,
            timeline=focus_timeline,
            timeline_truncated=focus_timeline_truncated,
        ),
    }


__all__ = ["build_trace_report", "function_payload"]
