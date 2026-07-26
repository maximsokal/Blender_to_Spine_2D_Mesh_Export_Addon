"""Focused tests for stable, dependency-scoped Rewrite readiness v2."""

from __future__ import annotations

from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1ExportReadinessReport,
    A1ObjectReadiness,
    A1ReadinessState,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    a1_export_readiness,
    a1_readiness_invalidation,
)


def _identity_matrix():
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
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
        matrix_world=_identity_matrix(),
        location=(0.0, 0.0, 0.0),
        rotation_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        hide_render=False,
        modifiers=(),
        spine2d_bake_settings=SimpleNamespace(
            bake_frame_start=0,
            frames_for_render=0,
        ),
        spine2d_connect_settings=SimpleNamespace(enabled=False),
    )


def _light_object(name: str, pointer: int):
    light_data = SimpleNamespace(
        id_type="LIGHT",
        name=f"{name}Data",
        name_full=f"{name}Data",
        as_pointer=lambda: pointer + 10_000,
    )
    return SimpleNamespace(
        id_type="OBJECT",
        type="LIGHT",
        name=name,
        name_full=name,
        as_pointer=lambda: pointer,
        data=light_data,
        material_slots=(),
        matrix_world=_identity_matrix(),
        location=(0.0, 0.0, 1.0),
        rotation_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        hide_render=False,
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
        render=SimpleNamespace(engine="BLENDER_EEVEE_NEXT"),
        frame_current=0,
        spine2d_texture_size=1024,
        spine2d_json_path="//exports",
        spine2d_images_path="images",
        spine2d_control_icons=True,
        spine2d_export_preview_animation=True,
        spine2d_seam_maker_mode="AUTO",
        spine2d_angle_limit=30.0,
        spine2d_angular_mode="SEED_CONE",
        spine2d_local_angle_limit=30.0,
        spine2d_frames_for_render=0,
        spine2d_bake_frame_start=0,
        spine2d_material_source_policy="REQUIRE_SOURCE",
        spine2d_generated_material_pattern="SOLID_GRAY",
        spine2d_projection_alpha_threshold=1.0 / 255.0,
    )
    return SimpleNamespace(
        scene=scene,
        active_object=source,
        selected_objects=(source,),
        view_layer=SimpleNamespace(update=lambda: None),
    )


def _report(
    *,
    signature: str = "fresh-signature",
    texture_pipeline: str | None = None,
) -> A1ExportReadinessReport:
    statistics = {}
    if texture_pipeline is not None:
        statistics["texture_pipeline"] = texture_pipeline
    return A1ExportReadinessReport(
        signature=signature,
        objects=(
            A1ObjectReadiness(
                object_id="Hero",
                statistics=statistics,
            ),
        ),
    )


def _entry(context):
    return a1_export_readiness._READINESS_CACHE[
        a1_export_readiness._scene_key(context.scene)
    ]


def _update(updated_id, **flags):
    return SimpleNamespace(id=updated_id, **flags)


def _depsgraph(*updates):
    return SimpleNamespace(updates=updates)


def _store(context, report=None):
    a1_readiness_invalidation.clear_a1_export_readiness()
    a1_readiness_invalidation.store_a1_export_readiness(
        context,
        report or _report(),
    )


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


def test_known_source_mesh_geometry_update_invalidates_with_precise_reason():
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
        reason = a1_readiness_invalidation.current_a1_readiness_reason(context)
        assert reason == "Object 'Hero' data geometry changed"
    finally:
        a1_readiness_invalidation.clear_a1_export_readiness()


def test_known_source_transform_update_invalidates_with_precise_reason():
    context = _context(451)
    try:
        _store(context)

        a1_readiness_invalidation.a1_readiness_depsgraph_update_post(
            context.scene,
            _depsgraph(
                _update(
                    context.active_object,
                    is_updated_geometry=False,
                    is_updated_transform=True,
                    is_updated_shading=False,
                )
            ),
        )

        assert _entry(context).stale is True
        reason = a1_readiness_invalidation.current_a1_readiness_reason(context)
        assert reason == "Object 'Hero' transform changed"
    finally:
        a1_readiness_invalidation.clear_a1_export_readiness()


def test_unrelated_non_temporary_blender_id_does_not_invalidate_report():
    context = _context(501)
    unrelated_mesh = SimpleNamespace(
        id_type="MESH",
        name="UnrelatedMesh",
        name_full="UnrelatedMesh",
        as_pointer=lambda: 999_501,
    )
    try:
        _store(context)

        a1_readiness_invalidation.a1_readiness_depsgraph_update_post(
            context.scene,
            _depsgraph(
                _update(
                    unrelated_mesh,
                    is_updated_geometry=True,
                    is_updated_transform=False,
                    is_updated_shading=False,
                )
            ),
        )

        assert _entry(context).stale is False
    finally:
        a1_readiness_invalidation.clear_a1_export_readiness()


def test_analysis_guard_ignores_even_known_dependency_updates():
    context = _context(551)
    key = a1_export_readiness._scene_key(context.scene)
    try:
        _store(context)
        a1_readiness_invalidation._ANALYSIS_SCENES.add(key)

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

        assert _entry(context).stale is False
    finally:
        a1_readiness_invalidation.clear_a1_export_readiness()


def test_analyse_signs_post_flush_state_instead_of_pre_analysis_state(monkeypatch):
    context = _context(601)
    state = {"signature": "before"}
    context.view_layer = SimpleNamespace(
        update=lambda: state.__setitem__("signature", "after")
    )
    monkeypatch.setattr(
        a1_readiness_invalidation,
        "_BASE_ANALYSE",
        lambda _context: _report(signature="pre-analysis-report"),
    )
    monkeypatch.setattr(
        a1_readiness_invalidation,
        "_BASE_BUILD_SIGNATURE",
        lambda _context: state["signature"],
    )

    report = a1_readiness_invalidation.analyse_a1_export_readiness(context)

    assert state["signature"] == "after"
    assert report.signature == a1_readiness_invalidation.build_a1_readiness_signature(
        context,
        report,
    )
    assert report.signature != "pre-analysis-report"


def test_stored_report_remains_current_after_its_own_flush(monkeypatch):
    context = _context(651)
    state = {"signature": "before"}
    context.view_layer = SimpleNamespace(
        update=lambda: state.__setitem__("signature", "after")
    )
    monkeypatch.setattr(
        a1_readiness_invalidation,
        "_BASE_BUILD_SIGNATURE",
        lambda _context: state["signature"],
    )
    monkeypatch.setattr(
        a1_export_readiness,
        "build_a1_readiness_signature",
        a1_readiness_invalidation.build_a1_readiness_signature,
    )
    try:
        _store(context, _report(signature="before"))

        state_value, current = a1_export_readiness.current_a1_export_readiness(context)

        assert state_value is A1ReadinessState.READY
        assert current is _entry(context).report
        assert current.signature == (
            a1_readiness_invalidation.build_a1_readiness_signature(context)
        )
    finally:
        a1_readiness_invalidation.clear_a1_export_readiness()


def test_camera_projection_signature_tracks_new_scene_light():
    context = _context(701)
    report = _report(texture_pipeline="CAMERA_PROJECTION")

    first = a1_readiness_invalidation.build_a1_readiness_signature(context, report)
    light = _light_object("Key", 777)
    context.scene.objects = context.scene.objects + (light,)
    second = a1_readiness_invalidation.build_a1_readiness_signature(context, report)

    assert first != second


def test_object_bake_signature_ignores_unrelated_scene_light():
    context = _context(751)
    report = _report(texture_pipeline="OBJECT_BAKE")

    first = a1_readiness_invalidation.build_a1_readiness_signature(context, report)
    light = _light_object("Irrelevant", 778)
    context.scene.objects = context.scene.objects + (light,)
    second = a1_readiness_invalidation.build_a1_readiness_signature(context, report)

    assert first == second


def test_signature_mismatch_reports_selection_scene_or_settings_reason(monkeypatch):
    context = _context(801)
    monkeypatch.setattr(
        a1_export_readiness,
        "build_a1_readiness_signature",
        a1_readiness_invalidation.build_a1_readiness_signature,
    )
    try:
        _store(context)
        context.scene.spine2d_texture_size = 2048

        state, _current = a1_export_readiness.current_a1_export_readiness(context)

        assert state is A1ReadinessState.STALE
        assert a1_readiness_invalidation.current_a1_readiness_reason(context) == (
            "Export selection, frame, Scene, or settings changed"
        )
    finally:
        a1_readiness_invalidation.clear_a1_export_readiness()
