"""Create a deterministic Mesh fixture used by the isolated parity harness smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import bpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.blender_headless.run_bake_integration import (  # noqa: E402
    _activate_only,
    _clear_scene,
    _create_emission_material,
    _create_quad,
)


def _arguments_after_separator() -> list[str]:
    try:
        index = sys.argv.index("--")
    except ValueError:
        return []
    return sys.argv[index + 1 :]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    namespace = parser.parse_args(_arguments_after_separator())
    output = namespace.output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)

    _clear_scene()
    source = _create_quad("HarnessHero")
    _create_emission_material(source)
    _activate_only(source)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 10
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    if not output.is_file():
        raise RuntimeError(f"Fixture .blend was not saved: {output}")
    print(f"Created fixture: {output}")


if __name__ == "__main__":
    main()
