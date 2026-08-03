"""Generate persistent 0.81.0 Depth Camera Projection acceptance artifacts.

Unlike the headless regression runners, this utility keeps every generated JSON and PNG
so a developer can inspect and import the exact test outputs after the automated gates.
The multi-object sample intentionally contains animated source and camera transforms;
only material animation may change its sequence PNGs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1MultiObjectExportSettings,
    A1MultiObjectMode,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    export_a1_multi_object,
)
from run_bake_integration import (  # noqa: E402
    _activate_only,
    _assert,
    _clear_scene,
    _create_sentinel,
)
from run_camera_projection_integration import (  # noqa: E402
    _configure_scene,
    _create_camera,
    _purge_orphan_scene_data,
)
from run_depth_camera_projection_integration import (  # noqa: E402
    _cases,
    _run_case,
)
from run_depth_camera_projection_multi_object_integration import (  # noqa: E402
    _keyframe_active_camera,
    _sources,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate persistent Depth Camera Projection sample outputs."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory that will receive all generated JSON, PNG, and manifest files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before generation when it already exists.",
    )
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    return parser.parse_args(argv)


def _prepare_multi_scene() -> None:
    _clear_scene()
    _purge_orphan_scene_data()
    _configure_scene()
    bpy.context.scene.cycles.samples = 1
    camera = _create_camera(name="DepthSamplesMultiCamera")
    camera.data.type = "PERSP"
    _keyframe_active_camera(camera)
    sentinel = _create_sentinel()
    sentinel.location = (8.0, 0.0, 0.0)
    _activate_only(sentinel)
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.update()


def _generate_multi_object(output_root: Path) -> tuple[Path, ...]:
    _prepare_multi_scene()
    output_root.mkdir(parents=True, exist_ok=False)
    sources = _sources(output_root)
    settings = A1MultiObjectExportSettings(
        output_directory=output_root,
        output_stem="DepthCameraProjectionMixedTiming",
        mode=A1MultiObjectMode.STANDALONE,
    )
    result = export_a1_multi_object(
        sources,
        settings,
        context=bpy.context,
        scene=bpy.context.scene,
    )
    _assert(result.success, f"Persistent multi-object export failed: {result.issues}")
    _assert(
        len(result.output_files) == 4,
        f"Persistent multi-object export expected JSON + 3 PNG: {result.output_files}",
    )
    return tuple(Path(path) for path in result.output_files)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_manifest(output_root: Path, elapsed_seconds: float) -> Path:
    files = tuple(
        sorted(
            path
            for path in output_root.rglob("*")
            if path.is_file() and path.name != "sample_manifest.json"
        )
    )
    manifest = {
        "generator": "generate_depth_camera_projection_samples.py",
        "extension_version": "0.81.0",
        "blender_version": ".".join(str(value) for value in bpy.app.version),
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "single_case_count": len(_cases()),
        "multi_object_case_count": 1,
        "multi_object_sequence_contract": (
            "material changes; source geometry and active camera remain frozen"
        ),
        "files": [
            {
                "path": path.relative_to(output_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
    }
    manifest_path = output_root / "sample_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    arguments = _arguments()
    output_root = arguments.output_root.expanduser().resolve(strict=False)
    if output_root.exists():
        if not arguments.clean:
            raise FileExistsError(
                f"Output root already exists: {output_root}. Use --clean to replace it."
            )
        if not output_root.is_dir():
            raise NotADirectoryError(f"Output root is not a directory: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    single_root = output_root / "single"
    single_root.mkdir(parents=True, exist_ok=False)
    cases = _cases()
    for index, case in enumerate(cases, start=1):
        print(f"[DEPTH-CAMERA-SAMPLES] RUN single {index}/{len(cases)} {case.key}")
        _run_case(single_root, case)
        print(f"[DEPTH-CAMERA-SAMPLES] PASS single {case.key}")

    multi_files = _generate_multi_object(output_root / "multi_object")
    print(
        "[DEPTH-CAMERA-SAMPLES] PASS multi_object "
        f"files={len(multi_files)}"
    )

    elapsed = time.perf_counter() - started
    manifest_path = _write_manifest(output_root, elapsed)
    physical_files = tuple(path for path in output_root.rglob("*") if path.is_file())
    print(
        "[DEPTH-CAMERA-SAMPLES] PASS "
        f"single_cases={len(cases)} multi_cases=1 files={len(physical_files)} "
        f"output={output_root} manifest={manifest_path} elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
