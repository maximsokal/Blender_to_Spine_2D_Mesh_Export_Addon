#!/usr/bin/env python3
"""Run Blender memory stress and evaluate the post-warmup RSS plateau."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_package import _resolve_blender_executable  # noqa: E402


WORKER = ROOT / "tools" / "blender_memory_stress.py"


class MemoryPlateauGateError(RuntimeError):
    """Raised when the Blender memory plateau gate cannot complete safely."""


def build_command(
    blender: str,
    fixture: Path,
    output_root: Path,
    report_json: Path,
    *,
    warmup: int,
    iterations: int,
    sample_every: int,
) -> list[str]:
    """Build one isolated Blender command for the memory-stress worker."""

    return [
        blender,
        "--background",
        "--factory-startup",
        "--debug-memory",
        "--log-show-memory",
        str(fixture),
        "--python-exit-code",
        "1",
        "--python",
        str(WORKER),
        "--",
        "--output-root",
        str(output_root),
        "--report-json",
        str(report_json),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--sample-every",
        str(sample_every),
    ]


def evaluate_plateau(
    payload: Mapping[str, Any],
    *,
    maximum_tail_growth_bytes: int,
    maximum_slope_bytes_per_sample: float,
) -> dict[str, int | float | bool]:
    """Evaluate the second half of RSS samples against growth and slope budgets."""

    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) < 3:
        raise MemoryPlateauGateError(
            "memory report must contain at least 3 RSS samples"
        )

    values: list[int] = []
    for index, item in enumerate(raw_samples):
        if not isinstance(item, Mapping):
            raise MemoryPlateauGateError(f"samples[{index}] must be an object")
        value = item.get("rss_bytes")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise MemoryPlateauGateError(
                f"samples[{index}].rss_bytes is invalid"
            )
        values.append(value)

    tail = values[len(values) // 2 :]
    growth = max(tail) - min(tail)
    x_mean = (len(tail) - 1) / 2.0
    y_mean = sum(tail) / len(tail)
    denominator = sum((x - x_mean) ** 2 for x in range(len(tail)))
    slope = (
        0.0
        if denominator == 0.0
        else sum(
            (x - x_mean) * (value - y_mean)
            for x, value in enumerate(tail)
        )
        / denominator
    )
    compatible = (
        growth <= maximum_tail_growth_bytes
        and slope <= maximum_slope_bytes_per_sample
    )

    return {
        "compatible": compatible,
        "tail_sample_count": len(tail),
        "tail_growth_bytes": growth,
        "slope_bytes_per_sample": slope,
        "maximum_tail_growth_bytes": maximum_tail_growth_bytes,
        "maximum_slope_bytes_per_sample": maximum_slope_bytes_per_sample,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blender",
        default=None,
        help=(
            "Path to Blender 5.2. Defaults to BLENDER_EXECUTABLE or PATH."
        ),
    )
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--max-tail-growth-mib", type=float, default=64.0)
    parser.add_argument("--max-slope-mib-per-sample", type=float, default=2.0)
    return parser


def _validate_arguments(
    *,
    warmup: int,
    iterations: int,
    sample_every: int,
    max_tail_growth_mib: float,
    max_slope_mib_per_sample: float,
) -> None:
    if warmup < 0:
        raise MemoryPlateauGateError("warmup must be non-negative")
    if iterations <= 0:
        raise MemoryPlateauGateError("iterations must be positive")
    if sample_every <= 0:
        raise MemoryPlateauGateError("sample-every must be positive")
    if max_tail_growth_mib < 0:
        raise MemoryPlateauGateError(
            "max-tail-growth-mib must be non-negative"
        )
    if max_slope_mib_per_sample < 0:
        raise MemoryPlateauGateError(
            "max-slope-mib-per-sample must be non-negative"
        )


def run(arguments: Sequence[str] | None = None) -> int:
    """Run one Blender memory process and persist its evaluation report."""

    args = _parser().parse_args(arguments)
    _validate_arguments(
        warmup=args.warmup,
        iterations=args.iterations,
        sample_every=args.sample_every,
        max_tail_growth_mib=args.max_tail_growth_mib,
        max_slope_mib_per_sample=args.max_slope_mib_per_sample,
    )

    explicit_blender = None if args.blender is None else Path(args.blender)
    blender = str(_resolve_blender_executable(explicit_blender))
    fixture = args.fixture.expanduser().resolve(strict=True)
    if not fixture.is_file() or fixture.suffix.casefold() != ".blend":
        raise MemoryPlateauGateError(
            f"fixture must be an existing .blend file: {fixture}"
        )

    work_root = args.work_root.expanduser().resolve(strict=False)
    work_root.mkdir(parents=True, exist_ok=True)
    worker_report = work_root / "memory-worker.json"
    blender_log = work_root / "blender-memory.log"

    command = build_command(
        blender,
        fixture,
        work_root / "outputs",
        worker_report,
        warmup=args.warmup,
        iterations=args.iterations,
        sample_every=args.sample_every,
    )

    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:
        raise MemoryPlateauGateError(
            f"unable to start Blender memory worker: {exc}"
        ) from exc

    blender_log.write_text(
        completed.stdout or "",
        encoding="utf-8",
        errors="replace",
        newline="\n",
    )

    if completed.returncode != 0:
        raise MemoryPlateauGateError(
            "Blender memory worker failed with "
            f"exit code {completed.returncode}; see {blender_log}"
        )
    if not worker_report.is_file():
        raise MemoryPlateauGateError(
            f"Blender memory worker did not create {worker_report}"
        )

    try:
        payload = json.loads(worker_report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MemoryPlateauGateError(
            f"unable to read Blender memory report {worker_report}: {exc}"
        ) from exc

    if not payload.get("success"):
        raise MemoryPlateauGateError(
            str(payload.get("error", "Blender memory worker failed"))
        )

    evaluation = evaluate_plateau(
        payload,
        maximum_tail_growth_bytes=int(
            args.max_tail_growth_mib * 1024 * 1024
        ),
        maximum_slope_bytes_per_sample=(
            args.max_slope_mib_per_sample * 1024 * 1024
        ),
    )
    final = {
        "worker": payload,
        "evaluation": evaluation,
        "command": command,
    }
    report_path = work_root / "memory-plateau-report.json"
    report_path.write_text(
        json.dumps(final, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0 if evaluation["compatible"] else 1


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
