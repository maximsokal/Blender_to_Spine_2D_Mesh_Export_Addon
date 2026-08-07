"""Run Normal/UV geometry and rig acceptance on real coin geometry.

The artist material in ``coin_star.blend`` is intentionally not part of these geometry
acceptance checks. It may require Camera/Depth because of displacement or other render
features; that capability is validated separately by
``run_coin_star_real_blend_shader_capability_integration.py``.

This runner temporarily assigns a deterministic camera-context surface material to the
real coin. Fresnel keeps the material on the same supported Normal/UV CAMERA_COMBINED
object-bake boundary exercised by the production coin gates, while the graph contains no
Volume or Displacement output. The three existing Normal/Active-Camera geometry and rig
gates then run unchanged. The original artist material is restored and the temporary
material datablock is removed in ``finally``; the source .blend is never saved.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

import bpy


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIRECTORY.parents[1]
for path in (SCRIPT_DIRECTORY, REPOSITORY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_object_analysis import (  # noqa: E402
    analyse_object_materials,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_object_audit import (  # noqa: E402
    audit_object_material_capabilities,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_routing import (  # noqa: E402
    _normal_uv_blocking_camera_findings,
    strongest_object_capability,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    ShaderBakeCapability,
)
from run_bake_integration import _assert  # noqa: E402
import run_coin_star_normal_camera_root_modes_integration as camera_root_gate  # noqa: E402
import run_coin_star_normal_object_root_setup_compensation_integration as object_root_gate  # noqa: E402
import run_coin_star_normal_projection_parity_integration as projection_gate  # noqa: E402
from run_coin_star_real_blend_shader_capability_integration import (  # noqa: E402
    _datablock_fingerprint,
    _object_fingerprint,
    _require_loaded_blend,
    _require_source_object,
    _scene_fingerprint,
)


_SAFE_MATERIAL_NAME = "Spine2D Publication Normal UV Material"


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description=(
            "Run real-coin Normal/UV geometry gates with a supported "
            "camera-context material override."
        )
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact coin_star.blend path Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _create_safe_material() -> bpy.types.Material:
    """Create a deterministic CAMERA_COMBINED-compatible material without displacement."""

    existing = bpy.data.materials.get(_SAFE_MATERIAL_NAME)
    _assert(
        existing is None,
        f"temporary publication material already exists: {_SAFE_MATERIAL_NAME}",
    )

    material = bpy.data.materials.new(name=_SAFE_MATERIAL_NAME)
    try:
        material.use_nodes = True
        node_tree = material.node_tree
        _assert(node_tree is not None, "temporary material has no node tree")

        nodes = node_tree.nodes
        nodes.clear()

        output = nodes.new(type="ShaderNodeOutputMaterial")
        diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
        fresnel = nodes.new(type="ShaderNodeFresnel")

        # Keep one genuine source/camera-context dependency so Normal/UV planning
        # exercises CAMERA_COMBINED rather than the simpler local DIFFUSE/EMIT routes.
        fresnel.inputs["IOR"].default_value = 1.45
        diffuse.inputs["Roughness"].default_value = 0.35
        node_tree.links.new(fresnel.outputs["Fac"], diffuse.inputs["Color"])
        node_tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])

        # No link is ever created to Material Output Volume or Displacement.
        return material
    except Exception:
        if material.users == 0:
            bpy.data.materials.remove(material)
        raise


def _assert_safe_material_route(source: bpy.types.Object) -> None:
    """Prove the override selects supported camera-context COMBINED object baking."""

    analysis = analyse_object_materials(
        source,
        source_object_id=source.name_full,
        render_target="CYCLES",
    )
    audits = audit_object_material_capabilities(
        source,
        analysis,
        render_target="CYCLES",
    )
    _assert(len(audits) == 1, f"safe material audit count is wrong: {len(audits)}")

    capability = strongest_object_capability(audits)
    _assert(
        capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        "safe publication material must exercise the CAMERA_COMBINED route: "
        f"{capability.value}",
    )

    findings = audits[0].findings
    _assert(
        any(
            finding.code == "SOURCE_OR_CAMERA_CONTEXT"
            and finding.node_type == "FRESNEL"
            for finding in findings
        ),
        f"safe publication material lost its Fresnel camera-context finding: {findings}",
    )

    blockers = _normal_uv_blocking_camera_findings(audits)
    _assert(
        not blockers,
        f"safe publication material has Normal/UV blockers: {blockers}",
    )


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    source = _require_source_object()
    slot = source.material_slots[0]
    original_material = slot.material
    _assert(original_material is not None, "real coin material slot became empty")

    scene_before = _scene_fingerprint()
    object_before = _object_fingerprint(source)
    datablocks_before = _datablock_fingerprint()
    safe_material: bpy.types.Material | None = None

    try:
        safe_material = _create_safe_material()
        slot.material = safe_material
        bpy.context.view_layer.update()
        _assert_safe_material_route(source)

        projection_gate._run(expected_blend)
        camera_root_gate._run(expected_blend)
        object_root_gate._run(expected_blend)
    finally:
        slot.material = original_material
        bpy.context.view_layer.update()
        if safe_material is not None:
            _assert(
                safe_material.users == 0,
                f"temporary publication material still has users: {safe_material.users}",
            )
            bpy.data.materials.remove(safe_material)

    _assert(
        _scene_fingerprint() == scene_before,
        "publication wrapper changed Blender context",
    )
    _assert(
        _object_fingerprint(source) == object_before,
        "publication wrapper changed source data",
    )
    _assert(
        _datablock_fingerprint() == datablocks_before,
        "publication wrapper leaked or removed Blender datablocks",
    )

    print(
        "[COIN-NORMAL-GEOMETRY-PUBLICATION] PASS "
        f"blend={loaded} object={source.name_full!r} "
        f"original_material={original_material.name_full!r} "
        "override=CAMERA_RENDER_REQUIRED/FRESNEL "
        "bake_route=CAMERA_COMBINED blockers=none "
        "gates=projection+camera-roots+object-root-setup source=restored",
        flush=True,
    )


def main() -> None:
    arguments = _parse_arguments()
    try:
        _run(arguments.expected_blend)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
