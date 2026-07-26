"""Focused tests for precise Rewrite readiness depsgraph invalidation."""

from __future__ import annotations

from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1ExportReadinessReport,
    A1ObjectReadiness,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    a1_export_readiness,
    a1_readiness_invalidation,
)


def _mesh_object(name: str, pointer: int):
    mesh = SimpleNamespace(
        id_type="MESH",
        name=f"{name}Mesh",
        name_full=f"{name}Mesh",
        as_pointer=lambda: pointer + 10_000,
        vertices=(),
        edges=(),
        loops=(),
        polygons=(),
    )
    return SimpleNamespace(
        id_type="OBJECT",
        type="MESH",
        name=name,
        name_full=name,
        as_pointer=lambda: pointer,
        data=mesh,
        material_slots=(),
    )


def _context(pointer: int = 100):
    source = _mesh_object("Hero", pointer)
    scene = SimpleNamespace(
        id_type="SCENE",
        name="Scene",
        name_full="Scene",
        as_pointer=lambda: pointer + 20_000,
        objects=(source,),
        world=None,
        node_tree=None,
        camera=None,
    )
    return SimpleNamespace(
        scene=scene,
        active_object=source,
        selected_objects=(source,),
    )


def _report() -> A1ExportReadinessReport:
    return A1ExportReadinessReport(
        signature="fresh-signature",
        objects=(A1ObjectReadiness(object_id="Hero"),),
    )


def _entry(context):
    return a1_export_readiness._READINESS_CACHE[
        a1_export_readiness._scene_key(context.scene)
    ]


def _update(updated_id, **flags):
    return SimpleNamespace(id=updated_id, **flags)


def _depsgraph(*updates):
    return SimpleNamespace(updates=updates)


def _store(context):
    a1_readiness_invalidation.clear_a1_export_readiness()
    a1_readiness_invalidation.store_a1_export_readiness(context, _report())


def test_empty_update_batch_does_not_invalidate_fresh_report():
    context = _context(101)
    try:
        _store(context)

        a1_readiness_invalidation.a1_readiness_depsgraph_update_post(
            context.scene,
            _depsgraph(),
        )

        assert _entry(context).stale is False
    finally:
        a1_readiness_invalidation.clear_a1_export_readiness()


def test_rewrite_temporary_datablock_update_is_ignored():
    context = _context(201)
    temporary_mesh = SimpleNamespace(
        id_type="MESH",
        name="__Spine2D_UnwrapMesh",
        name_full="__Spine2D_UnwrapMesh",
        as_pointer=lambda: 999_001,
    )
    try:
        _store(context)

        a1_readiness_invalidation.a1_readiness_depsgraph_update_post(
            context.scene,
            _depsgraph(
                _update(
                    temporary_mesh,
                    is_updated_geometry=True,
                    is_updated_transform=False,
                    is_updated_shading=False,
                )
            ),
        )

        assert _entry(context).stale is False
    finally:
        a1_readiness_invalidation.clear_a1_export_readiness()


def test_selection_only_known_object_update_is_ignored():
    context = _context(301)
    try:
        _store(context)

        a1_readiness_invalidation.a1_readiness_depsgraph_update_post(
            context.scene,
            _depsgraph(
                _update(
                    context.active_object,
                    is_updated_geometry=False,
                    is_updated_transform=False,
                    is_updated_shading=False,
                )
            ),
        )

        assert _entry(context).stale is False
    finally:
        a1_readiness_invalidation.clear_a1_export_readiness()


def test_known_source_mesh_geometry_update_invalidates_report():
    context = _context(401)
    try:
        _store(context)

        a1_readiness_invalidation.a1_readiness_depsgraph_update_post(
            context.scene,
            _depsgraph(
                _update(
                    context.active_object.data,
                    is_updated_geometry=True,
                    is_updated_transform=False,
                    is_updated_shading=False,
                )
            ),
        )

        assert _entry(context).stale is True
    finally:
        a1_readiness_invalidation.clear_a1_export_readiness()


def test_new_non_temporary_relevant_datablock_invalidates_report():
    context = _context(501)
    new_mesh = SimpleNamespace(
        id_type="MESH",
        name="NewMesh",
        name_full="NewMesh",
        as_pointer=lambda: 999_501,
    )
    try:
        _store(context)

        a1_readiness_invalidation.a1_readiness_depsgraph_update_post(
            context.scene,
            _depsgraph(
                _update(
                    new_mesh,
                    is_updated_geometry=True,
                    is_updated_transform=False,
                    is_updated_shading=False,
                )
            ),
        )

        assert _entry(context).stale is True
    finally:
        a1_readiness_invalidation.clear_a1_export_readiness()
