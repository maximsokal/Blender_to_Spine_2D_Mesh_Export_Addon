"""Real grenade.blend regression for Bump Only displacement through Normal/UV.

This gate intentionally uses the artist-authored file that exposed the production bug:
``E:/test_BtSe/grenade/grenade.blend``. It verifies the complete route, not just the live
capability boundary:

* object ``Cube`` and material ``1006`` must still exist in the fixture;
* Material Output displacement must be connected in immutable analysis;
* Blender's live material mode must be ``BUMP``;
* capability auditing must replace the blanket displacement blocker with
  ``DISPLACEMENT_BUMP_CONTEXT``;
* Normal/UV planning must select a CAMERA-scoped ``CAMERA_COMBINED`` pass;
* a real single-object export must produce non-empty Spine JSON and PNG output;
* source object/material/scene state and Blender datablock namespaces must be restored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import traceback

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.application import (  # noqa: E402
    A1SingleObjectExportSettings,
    A1SourceGeometryMode,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (  # noqa: E402
    analyse_object_materials,
    export_a1_single_object,
    prepare_a1_object,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_displacement import (  # noqa: E402
    DISPLACEMENT_BUMP_CONTEXT_CODE,
    DISPLACEMENT_RENDER_REQUIRED_CODE,
    material_displacement_method,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_object_audit import (  # noqa: E402
    audit_object_material_capabilities,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_routing import (  # noqa: E402
    _normal_uv_blocking_camera_findings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    A1TextureExportMode,
    BakeEvaluationScope,
    BakeExecutionSettings,
    BakeMode,
    BakeStrategyId,
    MaterialSemanticChannel,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.projection import (  # noqa: E402
    A1ProjectionDirection,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.rig_profiles import (  # noqa: E402
    A1RigSetupPoseMode,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.uv import UvUnwrapSettings  # noqa: E402
from run_bake_integration import PNG_SIGNATURE, _assert  # noqa: E402


_EXPECTED_OBJECT_NAME = "Cube"
_EXPECTED_MATERIAL_NAME = "1006"
_GENERATED_UV_LAYER = "SpineBakeUV"
_TEXTURE_SIZE = 256
_RENDER_TARGET = "CYCLES"


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Run real grenade.blend Bump Only displacement Normal/UV export."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact grenade.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _require_loaded_blend(expected_blend: str) -> Path:
    expected = _resolved_path(expected_blend)
    loaded_raw = str(getattr(bpy.data, "filepath", "") or "").strip()
    _assert(bool(loaded_raw), "Blender has no loaded .blend filepath")
    loaded = _resolved_path(loaded_raw)
    _assert(
        loaded == expected,
        f"wrong real fixture loaded: expected={expected}, actual={loaded}",
    )
    _assert(loaded.is_file(), f"loaded grenade fixture does not exist: {loaded}")
    return loaded


def _require_source_object():
    source = bpy.data.objects.get(_EXPECTED_OBJECT_NAME)
    _assert(source is not None, f"missing grenade object: {_EXPECTED_OBJECT_NAME!r}")
    _assert(source.type == "MESH", f"grenade object must be MESH, got {source.type!r}")
    _assert(len(source.data.vertices) > 0, "grenade Cube mesh has no vertices")
    return source


def _require_material(source):
    materials = tuple(
        slot.material
        for slot in source.material_slots
        if getattr(slot, "material", None) is not None
    )
    matches = tuple(
        material
        for material in materials
        if material.name_full == _EXPECTED_MATERIAL_NAME
    )
    _assert(
        len(matches) == 1,
        "grenade Cube must use exactly one material named '1006'; "
        f"materials={tuple(material.name_full for material in materials)!r}",
    )
    return matches[0]


def _matrix_fingerprint(matrix) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _source_fingerprint(source, scene) -> tuple:
    return (
        source.name_full,
        source.data.name_full,
        _matrix_fingerprint(source.matrix_world),
        tuple(
            None if slot.material is None else slot.material.name_full
            for slot in source.material_slots
        ),
        tuple(layer.name for layer in source.data.uv_layers),
        (
            None
            if source.data.uv_layers.active is None
            else source.data.uv_layers.active.name
        ),
        (
            None
            if scene.camera is None
            else scene.camera.name_full
        ),
        int(scene.frame_current),
        str(scene.render.engine),
    )


def _datablock_fingerprint() -> tuple:
    return (
        tuple(sorted(item.name_full for item in bpy.data.objects)),
        tuple(sorted(item.name_full for item in bpy.data.meshes)),
        tuple(sorted(item.name_full for item in bpy.data.materials)),
        tuple(sorted(item.name_full for item in bpy.data.images)),
    )


def _settings(output_directory: Path) -> A1SingleObjectExportSettings:
    if not isinstance(output_directory, Path):
        raise TypeError("output_directory must be pathlib.Path")
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=_TEXTURE_SIZE,
            texture_height=_TEXTURE_SIZE,
            output_directory=output_directory,
            images_relative_path="images",
            bake_margin=4,
        ),
        prefix="Grenade Cube",
        output_stem="Grenade_Cube_Bump_Normal_Z",
        json_output_stem="Grenade_Cube_Bump_Normal_Z",
        source_geometry_mode=A1SourceGeometryMode.ORIGINAL,
        uv=UvUnwrapSettings(layer_name=_GENERATED_UV_LAYER),
        diffuse_mode=BakeMode.DIFFUSE,
        procedural_mode=BakeMode.COMBINED,
        bake_execution=BakeExecutionSettings(
            render_engine="CYCLES",
            samples=1,
            texture_export_mode=A1TextureExportMode.NORMAL_UV_SEGMENTS,
        ),
        rig_setup_pose_mode=A1RigSetupPoseMode.PRESERVE_COMPOSITION,
        projection_direction=A1ProjectionDirection.POSITIVE_Z,
    )


def _assert_capability_route(source, material) -> tuple:
    method = material_displacement_method(material)
    _assert(
        method == "BUMP",
        "grenade material '1006' no longer uses Bump Only displacement; "
        f"actual={method!r}",
    )

    analysis = analyse_object_materials(
        source,
        source_object_id=source.name_full,
        render_target=_RENDER_TARGET,
    )
    slots = tuple(
        slot for slot in analysis.slots if slot.material_name == _EXPECTED_MATERIAL_NAME
    )
    _assert(
        len(slots) == 1,
        "immutable grenade analysis lost material '1006' or made it ambiguous",
    )
    slot = slots[0]
    _assert(
        MaterialSemanticChannel.DISPLACEMENT in slot.semantic_channels,
        "grenade material '1006' no longer exposes the DISPLACEMENT semantic channel",
    )

    audits = audit_object_material_capabilities(
        source,
        analysis,
        render_target=_RENDER_TARGET,
    )
    matching_audits = tuple(
        audit for audit in audits if audit.material_name == _EXPECTED_MATERIAL_NAME
    )
    _assert(
        len(matching_audits) == 1,
        "grenade capability audit lost material '1006' or made it ambiguous",
    )
    audit = matching_audits[0]
    codes = tuple(finding.code for finding in audit.findings)
    _assert(
        DISPLACEMENT_BUMP_CONTEXT_CODE in codes,
        f"Bump Only displacement was not refined to bump context: {codes!r}",
    )
    _assert(
        DISPLACEMENT_RENDER_REQUIRED_CODE not in codes,
        f"blanket displacement blocker survived Bump Only refinement: {codes!r}",
    )
    blockers = _normal_uv_blocking_camera_findings(audits)
    _assert(
        blockers == (),
        f"grenade Bump Only material still blocks Normal/UV: {blockers!r}",
    )
    return analysis, audits


def _assert_prepared_plan(prepared) -> None:
    passes = tuple(prepared.bake_plan.passes)
    camera_combined = tuple(
        item
        for item in passes
        if item.strategy_id is BakeStrategyId.CAMERA_COMBINED
    )
    _assert(
        camera_combined,
        "grenade Bump Only preparation did not produce CAMERA_COMBINED",
    )
    for item in camera_combined:
        _assert(
            item.bake_mode is BakeMode.COMBINED,
            f"CAMERA_COMBINED pass uses wrong bake mode: {item.bake_mode}",
        )
        _assert(
            item.evaluation_scope is BakeEvaluationScope.CAMERA,
            f"CAMERA_COMBINED pass uses wrong scope: {item.evaluation_scope}",
        )


def _assert_outputs(result) -> tuple[Path, Path]:
    _assert(
        bool(result.success),
        "real grenade Normal/UV export failed: "
        f"issues={result.issues!r}, statistics={dict(result.statistics)!r}",
    )
    outputs = tuple(Path(path).resolve(strict=False) for path in result.output_files)
    json_files = tuple(path for path in outputs if path.suffix.lower() == ".json")
    png_files = tuple(path for path in outputs if path.suffix.lower() == ".png")
    _assert(len(json_files) == 1, f"expected one grenade JSON, got {json_files!r}")
    _assert(len(png_files) == 1, f"expected one grenade PNG, got {png_files!r}")
    _assert(
        all(path.is_file() and path.stat().st_size > 8 for path in outputs),
        f"grenade export contains missing or empty outputs: {outputs!r}",
    )
    _assert(
        png_files[0].read_bytes().startswith(PNG_SIGNATURE),
        f"grenade texture is not a PNG: {png_files[0]}",
    )
    document = json.loads(json_files[0].read_text(encoding="utf-8"))
    _assert(isinstance(document, dict), "grenade Spine JSON root must be a mapping")
    _assert(bool(document.get("bones")), "grenade Spine JSON contains no bones")
    _assert(bool(document.get("skins")), "grenade Spine JSON contains no skins")
    return json_files[0], png_files[0]


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    scene = bpy.context.scene
    source = _require_source_object()
    material = _require_material(source)
    _assert(scene.camera is not None, "grenade fixture must provide an active scene camera")

    source_before = _source_fingerprint(source, scene)
    datablocks_before = _datablock_fingerprint()
    _assert_capability_route(source, material)

    with tempfile.TemporaryDirectory(prefix="spine2d_grenade_bump_") as temp_root:
        output_directory = Path(temp_root).resolve(strict=False)
        settings = _settings(output_directory)

        prepared = prepare_a1_object(
            source,
            settings,
            context=bpy.context,
            scene=scene,
        )
        _assert_prepared_plan(prepared)

        result = export_a1_single_object(
            source,
            settings,
            context=bpy.context,
            scene=scene,
        )
        json_path, png_path = _assert_outputs(result)

        _assert(
            _source_fingerprint(source, scene) == source_before,
            "grenade export changed source object/material/scene state",
        )
        _assert(
            _datablock_fingerprint() == datablocks_before,
            "grenade export leaked or removed Blender datablocks",
        )

        print(
            "[GRENADE-BUMP-NORMAL-UV] PASS "
            f"blend={loaded} object={source.name_full!r} "
            f"material={material.name_full!r} displacement_method=BUMP "
            "semantic=DISPLACEMENT capability=DISPLACEMENT_BUMP_CONTEXT "
            "plan=CAMERA_COMBINED scope=CAMERA mode=COMBINED "
            f"json_bytes={json_path.stat().st_size} "
            f"png_bytes={png_path.stat().st_size} source=unchanged",
            flush=True,
        )


def main() -> None:
    arguments = _parse_arguments()
    _run(arguments.expected_blend)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
