"""Audit the real coin asset without mutating its Blender scene or shader graph.

Blender must open the exact caller-provided ``coin_star.blend`` before this script runs.
The regression reproduces the BlendKit ``Gold coin`` material that contains a muted
``Add Shader`` with two same-named inputs. Recursive traversal conservatively visits all
inputs, so that advisory must not mask the material's genuine camera-render requirement.
Fresnel and Generated coordinates remain reproducible by the source-object bake context;
the two Glossy BSDF nodes are the concrete Normal-mode blockers.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback
from typing import Any

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
    normal_mode_camera_requirement_message,
    strongest_object_capability,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    ShaderBakeCapability,
)
from run_bake_integration import _assert  # noqa: E402


_EXPECTED_OBJECT_NAME = "Game Gold Coin"
_EXPECTED_MATERIAL_NAME = "Gold coin"
_RENDER_TARGET = "CYCLES"
_MUTED_ADVISORY = (
    "Muted node 'Add Shader' has no unambiguous internal bypass for output "
    "'Shader'; all inputs were analyzed conservatively"
)


def _parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Audit the real coin shader capability boundary."
    )
    parser.add_argument(
        "--expected-blend",
        required=True,
        help="Exact coin_star.blend path that Blender must already have loaded.",
    )
    return parser.parse_args(arguments)


def _canonical_path(value: str | Path) -> str:
    return os.path.normcase(
        os.path.normpath(str(Path(value).expanduser().resolve(strict=False)))
    )


def _library_path(value: Any) -> str | None:
    library = getattr(value, "library", None)
    if library is None:
        return None
    filepath = str(getattr(library, "filepath", "") or "")
    if not filepath:
        return ""
    return _canonical_path(bpy.path.abspath(filepath))


def _socket_fingerprint(socket: Any) -> tuple[str, str, bool]:
    return (
        str(getattr(socket, "name", "") or ""),
        str(getattr(socket, "identifier", "") or ""),
        bool(getattr(socket, "is_linked", False)),
    )


def _material_fingerprint(material: Any) -> tuple[Any, ...]:
    node_tree = getattr(material, "node_tree", None)
    if node_tree is None:
        return (
            material.name_full,
            int(material.as_pointer()),
            _library_path(material),
            None,
        )

    nodes = tuple(
        sorted(
            (
                node.name,
                node.type,
                bool(node.mute),
                bool(getattr(node, "is_active_output", False)),
                int(node.as_pointer()),
                tuple(_socket_fingerprint(socket) for socket in node.inputs),
                tuple(_socket_fingerprint(socket) for socket in node.outputs),
                (
                    int(node.node_tree.as_pointer())
                    if getattr(node, "node_tree", None) is not None
                    else None
                ),
            )
            for node in node_tree.nodes
        )
    )
    links = tuple(
        sorted(
            (
                link.from_node.name,
                link.from_socket.name,
                link.to_node.name,
                link.to_socket.name,
            )
            for link in node_tree.links
        )
    )
    return (
        material.name_full,
        int(material.as_pointer()),
        _library_path(material),
        node_tree.name_full,
        int(node_tree.as_pointer()),
        nodes,
        links,
    )


def _object_fingerprint(source: bpy.types.Object) -> tuple[Any, ...]:
    return (
        source.name_full,
        source.type,
        int(source.as_pointer()),
        _library_path(source),
        source.data.name_full,
        int(source.data.as_pointer()),
        _library_path(source.data),
        tuple(tuple(float(value) for value in row) for row in source.matrix_world),
        tuple(
            (
                modifier.name,
                modifier.type,
                bool(modifier.show_viewport),
                bool(modifier.show_render),
            )
            for modifier in source.modifiers
        ),
        tuple(
            (
                slot.name,
                (
                    _material_fingerprint(slot.material)
                    if slot.material is not None
                    else None
                ),
            )
            for slot in source.material_slots
        ),
    )


def _scene_fingerprint() -> tuple[Any, ...]:
    active = bpy.context.view_layer.objects.active
    selected = tuple(sorted(obj.name_full for obj in bpy.context.selected_objects))
    camera = bpy.context.scene.camera
    return (
        int(bpy.context.scene.frame_current),
        str(bpy.context.scene.render.engine),
        camera.name_full if camera is not None else None,
        int(camera.as_pointer()) if camera is not None else None,
        active.name_full if active is not None else None,
        selected,
        str(bpy.context.mode),
    )


def _datablock_fingerprint() -> tuple[tuple[str, tuple[str, ...]], ...]:
    collections = (
        ("objects", bpy.data.objects),
        ("meshes", bpy.data.meshes),
        ("materials", bpy.data.materials),
        ("node_groups", bpy.data.node_groups),
        ("images", bpy.data.images),
    )
    return tuple(
        (label, tuple(sorted(item.name_full for item in collection)))
        for label, collection in collections
    )


def _require_loaded_blend(expected_blend: str) -> str:
    loaded = str(Path(bpy.data.filepath).resolve(strict=False))
    _assert(bool(loaded), "Blender has no loaded .blend path")
    _assert(
        _canonical_path(loaded) == _canonical_path(expected_blend),
        f"wrong real blend loaded: actual={loaded}, expected={expected_blend}",
    )
    return loaded


def _require_source_object() -> bpy.types.Object:
    source = bpy.data.objects.get(_EXPECTED_OBJECT_NAME)
    _assert(source is not None, f"missing real object: {_EXPECTED_OBJECT_NAME}")
    _assert(source.type == "MESH", f"{_EXPECTED_OBJECT_NAME} is not a MESH object")
    _assert(
        len(source.material_slots) == 1,
        f"real coin must have one material slot, got {len(source.material_slots)}",
    )
    material = source.material_slots[0].material
    _assert(material is not None, "real coin material slot is empty")
    _assert(
        material.name_full == _EXPECTED_MATERIAL_NAME,
        f"unexpected real coin material: {material.name_full}",
    )
    return source


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    source = _require_source_object()

    scene_before = _scene_fingerprint()
    object_before = _object_fingerprint(source)
    datablocks_before = _datablock_fingerprint()

    analysis = analyse_object_materials(
        source,
        source_object_id=source.name_full,
        render_target=_RENDER_TARGET,
    )
    _assert(len(analysis.slots) == 1, "coin material analysis lost its dense slot")
    graph = analysis.slots[0].graph
    _assert(graph is not None, "coin material analysis produced no graph snapshot")
    _assert(
        _MUTED_ADVISORY in graph.issues,
        f"real muted Add Shader advisory disappeared: {graph.issues}",
    )

    audits = audit_object_material_capabilities(
        source,
        analysis,
        render_target=_RENDER_TARGET,
    )
    _assert(len(audits) == 1, f"unexpected coin capability audit count: {len(audits)}")
    capability = strongest_object_capability(audits)
    _assert(
        capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        f"real coin capability is {capability.value}, expected CAMERA_RENDER_REQUIRED",
    )

    findings = audits[0].findings
    unsupported = tuple(
        finding
        for finding in findings
        if finding.capability is ShaderBakeCapability.UNSUPPORTED
    )
    _assert(
        not unsupported,
        f"conservatively analysed muted node remained unsupported: {unsupported}",
    )
    _assert(
        not any(finding.code == "GRAPH_ANALYSIS_INCOMPLETE" for finding in findings),
        "muted Add Shader advisory became GRAPH_ANALYSIS_INCOMPLETE",
    )
    _assert(
        any(finding.code == "GRAPH_CAMERA_DEPENDENCY" for finding in findings),
        "real coin lost its aggregate camera dependency",
    )
    _assert(
        any(
            finding.code == "SOURCE_OR_CAMERA_CONTEXT"
            and finding.node_type == "FRESNEL"
            for finding in findings
        ),
        "real coin lost the Fresnel camera-context finding",
    )
    glossy_findings = tuple(
        finding
        for finding in findings
        if finding.code == "SOURCE_OR_CAMERA_CONTEXT"
        and finding.node_type == "BSDF_GLOSSY"
    )
    _assert(
        len(glossy_findings) == 2,
        f"expected two Glossy camera findings, got {glossy_findings}",
    )
    _assert(
        any(
            finding.code == "TEXTURE_COORD_SOURCE_CONTEXT"
            and finding.node_type == "TEX_COORD"
            and finding.output_socket == "Generated"
            for finding in findings
        ),
        "real coin lost the Generated-coordinate camera-context finding",
    )

    guidance = normal_mode_camera_requirement_message(audits)
    _assert(
        "Camera Projection or Depth Camera Projection" in guidance,
        f"Normal-mode guidance does not identify supported camera routes: {guidance}",
    )
    _assert(
        guidance.count("BSDF_GLOSSY") == 2,
        f"Normal-mode guidance must expose both Glossy blockers exactly once: {guidance}",
    )
    _assert(
        "Generated" not in guidance,
        f"Normal-mode guidance misclassified supported Generated coordinates: {guidance}",
    )
    _assert(
        "FRESNEL" not in guidance,
        f"Normal-mode guidance misclassified supported Fresnel context: {guidance}",
    )
    _assert(
        "GRAPH_CAMERA_DEPENDENCY" not in guidance,
        f"Normal-mode guidance duplicated the aggregate camera finding: {guidance}",
    )

    _assert(_scene_fingerprint() == scene_before, "coin audit changed Blender context")
    _assert(_object_fingerprint(source) == object_before, "coin audit changed source data")
    _assert(
        _datablock_fingerprint() == datablocks_before,
        "coin audit created or removed Blender datablocks",
    )

    print(
        "[COIN-REAL-SHADER-CAPABILITY] PASS "
        f"blend={loaded} object={source.name_full!r} material={_EXPECTED_MATERIAL_NAME!r} "
        f"nodes={len(graph.reachable_nodes)} links={len(graph.reachable_links)} "
        f"findings={len(findings)} capability={capability.value} "
        "muted_fallback=advisory source_context=FRESNEL+Generated "
        "blockers=2xBSDF_GLOSSY normal_mode=camera-required scene=unchanged",
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
