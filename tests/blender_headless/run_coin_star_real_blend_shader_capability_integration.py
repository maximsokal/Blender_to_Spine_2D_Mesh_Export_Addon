"""Audit the real coin material against the public execution boundaries.

Blender must open the exact caller-provided ``coin_star.blend`` before this script runs.
The artist material is allowed to evolve: display names and exact node topology are not
fixture identity. The gate instead proves that the current graph is completely analyzable,
that no GROUP/UNSUPPORTED boundary is required, that Normal/UV blockers are reported
honestly, and that Camera/Depth remains a valid public route for CAMERA_RENDER_REQUIRED
materials. The audit must not mutate the Blender scene, object, or material graph.
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
    _normal_uv_blocking_camera_findings,
    strongest_object_capability,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (  # noqa: E402
    ShaderBakeCapability,
)
from run_bake_integration import _assert  # noqa: E402


_EXPECTED_OBJECT_NAME = "Game Gold Coin"
_RENDER_TARGET = "CYCLES"
_PUBLICLY_ROUTABLE_CAPABILITIES = frozenset(
    {
        ShaderBakeCapability.LOCAL_UV_SAFE,
        ShaderBakeCapability.SCENE_UV_SAFE,
        ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
    }
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
    """Resolve the real coin by stable object identity, not material display name."""

    source = bpy.data.objects.get(_EXPECTED_OBJECT_NAME)
    _assert(source is not None, f"missing real object: {_EXPECTED_OBJECT_NAME}")
    _assert(source.type == "MESH", f"{_EXPECTED_OBJECT_NAME} is not a MESH object")
    _assert(
        len(source.material_slots) == 1,
        f"real coin must have one material slot, got {len(source.material_slots)}",
    )
    material = source.material_slots[0].material
    _assert(material is not None, "real coin material slot is empty")
    return source


def _blocking_codes(
    blockers: tuple[tuple[str, tuple[tuple[str, str | None, str | None], ...]], ...],
) -> tuple[str, ...]:
    """Flatten deterministic Normal/UV blocker diagnostics for publication logs."""

    codes = tuple(
        code
        for _material_name, findings in blockers
        for code, _node_type, _output_socket in findings
    )
    _assert(
        all(isinstance(code, str) and bool(code.strip()) for code in codes),
        f"real coin blocker diagnostics contain an invalid code: {blockers}",
    )
    return codes


def _run(expected_blend: str) -> None:
    loaded = _require_loaded_blend(expected_blend)
    source = _require_source_object()
    material = source.material_slots[0].material
    _assert(material is not None, "real coin material slot became empty")

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

    audits = audit_object_material_capabilities(
        source,
        analysis,
        render_target=_RENDER_TARGET,
    )
    _assert(
        len(audits) == 1,
        f"unexpected coin capability audit count: {len(audits)}",
    )

    findings = audits[0].findings
    unsupported = tuple(
        finding
        for finding in findings
        if finding.capability is ShaderBakeCapability.UNSUPPORTED
    )
    _assert(
        not unsupported,
        f"real coin contains unsupported shader findings: {unsupported}",
    )
    incomplete = tuple(
        finding
        for finding in findings
        if finding.code == "GRAPH_ANALYSIS_INCOMPLETE"
    )
    _assert(
        not incomplete,
        f"real coin shader graph analysis is incomplete: {incomplete}",
    )

    capability = strongest_object_capability(audits)
    _assert(
        capability in _PUBLICLY_ROUTABLE_CAPABILITIES,
        "real coin requires an unsupported public execution boundary: "
        f"{capability.value}",
    )

    blockers = _normal_uv_blocking_camera_findings(audits)
    blocker_codes = _blocking_codes(blockers)
    if blocker_codes:
        _assert(
            capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
            "Normal/UV blockers are only valid for CAMERA_RENDER_REQUIRED material: "
            f"capability={capability.value}, blockers={blockers}",
        )
        normal_route = "blocked"
        camera_route = "supported"
    else:
        normal_route = "supported"
        camera_route = "supported"

    _assert(_scene_fingerprint() == scene_before, "coin audit changed Blender context")
    _assert(_object_fingerprint(source) == object_before, "coin audit changed source data")
    _assert(
        _datablock_fingerprint() == datablocks_before,
        "coin audit created or removed Blender datablocks",
    )

    print(
        "[COIN-REAL-SHADER-CAPABILITY] PASS "
        f"blend={loaded} object={source.name_full!r} material={material.name_full!r} "
        f"nodes={len(graph.reachable_nodes)} links={len(graph.reachable_links)} "
        f"graph_issues={len(graph.issues)} findings={len(findings)} "
        f"capability={capability.value} normal_mode={normal_route} "
        f"camera_modes={camera_route} blockers={blocker_codes} "
        "scene=unchanged",
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
