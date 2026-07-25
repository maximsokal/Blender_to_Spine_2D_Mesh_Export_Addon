#!/usr/bin/env python3
"""Generate deterministic public .blend fixtures in Blender 5.2."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.prepare_package import _resolve_blender_executable  # noqa: E402


WORKER = ROOT / "tools" / "create_public_blend_fixtures.py"
EXPECTED_FIXTURES = (
    "procedural_noise.blend",
    "nested_node_groups.blend",
    "overlapping_uv.blend",
    "non_manifold.blend",
    "negative_scale_modifier.blend",
)


class PublicFixtureGenerationError(RuntimeError):
    """Raised when Blender cannot generate the complete public fixture set."""


def build_command(
    blender: str,
    output_root: Path,
) -> list[str]:
    """Build one isolated Blender command for public fixture generation."""

    return [
        blender,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(WORKER),
        "--",
        "--output",
        str(output_root),
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blender",
        default=None,
        help=(
            "Path to Blender 5.2. Defaults to BLENDER_EXECUTABLE or PATH."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def run(arguments: Sequence[str] | None = None) -> int:
    """Generate all fixtures and fail closed on missing or empty files."""

    args = _parser().parse_args(arguments)
    explicit_blender = None if args.blender is None else Path(args.blender)
    blender = str(_resolve_blender_executable(explicit_blender))
    output_root = args.output_root.expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)

    command = build_command(blender, output_root)
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
        raise PublicFixtureGenerationError(
            f"unable to start Blender fixture worker: {exc}"
        ) from exc

    if completed.stdout:
        print(completed.stdout, end="")

    if completed.returncode != 0:
        raise PublicFixtureGenerationError(
            "Blender fixture generation failed with "
            f"exit code {completed.returncode}"
        )

    missing: list[str] = []
    empty: list[str] = []
    for name in EXPECTED_FIXTURES:
        path = output_root / name
        if not path.is_file():
            missing.append(name)
        elif path.stat().st_size <= 1024:
            empty.append(name)

    if missing or empty:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if empty:
            details.append("too small: " + ", ".join(empty))
        raise PublicFixtureGenerationError(
            "public fixture verification failed; " + "; ".join(details)
        )

    for name in EXPECTED_FIXTURES:
        print(output_root / name)
    return 0


def main() -> None:
    raise SystemExit(run())


if __name__ == "__main__":
    main()
