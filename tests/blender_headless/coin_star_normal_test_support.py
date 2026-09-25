"""Shared real-coin Normal/UV test support.

The artist-authored ``Gold metal`` material currently contains true displacement and is
therefore intentionally rejected by the production Normal / UV Segments route. Geometry,
projection, and rig regression tests still need to exercise that route against the real
coin mesh, so they temporarily replace only the material slot with a deterministic
camera-context surface material that has no Volume or Displacement output.

The context manager owns the entire mutation lifecycle and always restores the artist
material before returning to the caller. The source ``.blend`` is never saved.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator

import bpy

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_object_analysis import (
    analyse_object_materials,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_object_audit import (
    audit_object_material_capabilities,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_routing import (
    _normal_uv_blocking_camera_findings,
    strongest_object_capability,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import ShaderBakeCapability
from run_bake_integration import _assert


_SAFE_MATERIAL_NAME = "Spine2D Real Coin Normal UV Test Material"
_RENDER_TARGET = "CYCLES"


def _create_safe_normal_material() -> bpy.types.Material:
    """Create one deterministic camera-context surface material for Normal/UV gates."""

    existing = bpy.data.materials.get(_SAFE_MATERIAL_NAME)
    _assert(
        existing is None,
        f"temporary Normal/UV material already exists: {_SAFE_MATERIAL_NAME}",
    )

    material = bpy.data.materials.new(name=_SAFE_MATERIAL_NAME)
    try:
        material.use_nodes = True
        node_tree = material.node_tree
        _assert(node_tree is not None, "temporary Normal/UV material has no node tree")

        nodes = node_tree.nodes
        nodes.clear()

        output = nodes.new(type="ShaderNodeOutputMaterial")
        diffuse = nodes.new(type="ShaderNodeBsdfDiffuse")
        fresnel = nodes.new(type="ShaderNodeFresnel")

        fresnel.inputs["IOR"].default_value = 1.45
        diffuse.inputs["Roughness"].default_value = 0.35
        node_tree.links.new(fresnel.outputs["Fac"], diffuse.inputs["Color"])
        node_tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
        return material
    except Exception:
        if material.users == 0:
            bpy.data.materials.remove(material)
        raise


def _assert_safe_normal_route(source: bpy.types.Object) -> None:
    """Prove the temporary material exercises the supported camera-surface route."""

    if not isinstance(source, bpy.types.Object):
        raise TypeError("source must be bpy.types.Object")
    if source.type != "MESH" or source.data is None:
        raise TypeError("source must be a mesh object with data")

    analysis = analyse_object_materials(
        source,
        source_object_id=source.name_full,
        render_target=_RENDER_TARGET,
    )
    audits = audit_object_material_capabilities(
        source,
        analysis,
        render_target=_RENDER_TARGET,
    )
    _assert(
        len(audits) == 1,
        f"temporary Normal/UV material audit count is wrong: {len(audits)}",
    )
    _assert(
        strongest_object_capability(audits)
        is ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        "temporary Normal/UV material must exercise camera-context surface baking",
    )
    _assert(
        any(
            finding.code == "SOURCE_OR_CAMERA_CONTEXT"
            and finding.node_type == "FRESNEL"
            for finding in audits[0].findings
        ),
        f"temporary material lost Fresnel camera-context finding: {audits[0].findings}",
    )
    blockers = _normal_uv_blocking_camera_findings(audits)
    _assert(
        not blockers,
        f"temporary Normal/UV material unexpectedly has blockers: {blockers}",
    )


@contextmanager
def safe_coin_normal_material(source: bpy.types.Object) -> Iterator[bpy.types.Material]:
    """Temporarily replace the real coin artist material and restore it reliably."""

    if not isinstance(source, bpy.types.Object):
        raise TypeError("source must be bpy.types.Object")
    if source.type != "MESH" or source.data is None:
        raise TypeError("source must be a mesh object with data")
    if len(source.material_slots) != 1:
        raise ValueError(
            f"real coin fixture must have exactly one material slot; actual={len(source.material_slots)}"
        )

    slot = source.material_slots[0]
    original_material = slot.material
    _assert(original_material is not None, "real coin material slot is empty")

    safe_material: bpy.types.Material | None = None
    try:
        safe_material = _create_safe_normal_material()
        slot.material = safe_material
        bpy.context.view_layer.update()
        _assert_safe_normal_route(source)
        yield safe_material
    finally:
        slot.material = original_material
        bpy.context.view_layer.update()
        if safe_material is not None:
            _assert(
                safe_material.users == 0,
                f"temporary Normal/UV material still has users: {safe_material.users}",
            )
            bpy.data.materials.remove(safe_material)
