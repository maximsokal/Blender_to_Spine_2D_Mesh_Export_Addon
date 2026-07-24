"""Focused tests for typed A1 readiness state and Blender-session caching."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1ExportReadinessReport,
    A1ObjectReadiness,
    A1ReadinessState,
    ExportIssue,
    IssueSeverity,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import a1_export_readiness


def _issue(
    severity: IssueSeverity,
    *,
    code: str,
    object_id: str | None = None,
) -> ExportIssue:
    return ExportIssue(
        severity=severity,
        stage="TEST",
        code=code,
        message=code,
        object_id=object_id,
    )


def _report(
    *,
    signature: str = "signature",
    object_issues=(),
    global_issues=(),
) -> A1ExportReadinessReport:
    return A1ExportReadinessReport(
        signature=signature,
        objects=(
            A1ObjectReadiness(
                object_id="Mesh",
                issues=tuple(object_issues),
                statistics={"source_vertices": 4},
            ),
        ),
        issues=tuple(global_issues),
        statistics={"object_count": 1},
    )


def _mesh(name: str, pointer: int):
    data = SimpleNamespace(
        as_pointer=lambda: pointer + 10_000,
        name=f"{name}Mesh",
        name_full=f"{name}Mesh",
        vertices=(),
        edges=(),
        loops=(),
        polygons=(),
    )
    identity = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    return SimpleNamespace(
        type="MESH",
        name=name,
        name_full=name,
        as_pointer=lambda: pointer,
        data=data,
        matrix_world=identity,
        location=(0.0, 0.0, 0.0),
        rotation_euler=(0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        hide_render=False,
        modifiers=(),
        material_slots=(),
        spine2d_bake_settings=SimpleNamespace(
            bake_frame_start=0,
            frames_for_render=0,
        ),
        spine2d_connect_settings=SimpleNamespace(enabled=False),
    )


def _scene(pointer: int = 900):
    return SimpleNamespace(
        as_pointer=lambda: pointer,
        render=SimpleNamespace(engine="BLENDER_EEVEE_NEXT"),
        camera=None,
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


def test_readiness_state_uses_errors_as_blockers_and_warnings_as_non_blocking():
    ready = _report()
    warning = _report(
        object_issues=(_issue(IssueSeverity.WARNING, code="WARN", object_id="Mesh"),)
    )
    blocked = _report(
        global_issues=(_issue(IssueSeverity.ERROR, code="BLOCK"),)
    )

    assert ready.state is A1ReadinessState.READY
    assert ready.can_export
    assert warning.state is A1ReadinessState.WARNING
    assert warning.can_export
    assert warning.warning_count == 1
    assert blocked.state is A1ReadinessState.BLOCKED
    assert not blocked.can_export
    assert blocked.blocker_count == 1


def test_object_readiness_rejects_foreign_issue_and_invalid_statistics():
    with pytest.raises(ValueError, match="match object_id"):
        A1ObjectReadiness(
            object_id="Mesh",
            issues=(_issue(IssueSeverity.WARNING, code="FOREIGN", object_id="Other"),),
        )
    with pytest.raises(TypeError, match="finite float"):
        A1ObjectReadiness(object_id="Mesh", statistics={"flag": True})
    with pytest.raises(ValueError, match="finite"):
        A1ObjectReadiness(object_id="Mesh", statistics={"value": float("nan")})


def test_readiness_report_rejects_duplicate_object_ids():
    item = A1ObjectReadiness(object_id="Mesh")

    with pytest.raises(ValueError, match="unique object_id"):
        A1ExportReadinessReport(
            signature="signature",
            objects=(item, item),
        )


def test_single_request_identity_uses_active_mesh_not_unrelated_selection():
    selected = _mesh("Selected", 101)
    active = _mesh("Active", 102)
    context = SimpleNamespace(
        selected_objects=(selected,),
        active_object=active,
    )

    assert a1_export_readiness._request_mesh_objects(context) == (active,)
    assert a1_export_readiness._requested_object_ids(context) == ("Active",)


def test_selected_mesh_wrappers_are_deduplicated_before_mode_selection():
    first = _mesh("Mesh", 201)
    duplicate_wrapper = _mesh("Mesh", 201)
    context = SimpleNamespace(
        selected_objects=(first, duplicate_wrapper),
        active_object=first,
    )

    assert a1_export_readiness._selected_meshes(context) == (first,)
    assert a1_export_readiness._request_mesh_objects(context) == (first,)


def test_multi_request_uses_same_deterministic_order_as_ui_plan():
    first = _mesh("First", 301)
    second = _mesh("Second", 302)
    active_wrapper = _mesh("First", 301)
    context = SimpleNamespace(
        selected_objects=(second, first),
        active_object=active_wrapper,
    )

    assert a1_export_readiness._request_mesh_objects(context) == (first, second)
    assert a1_export_readiness._requested_object_ids(context) == (
        "First",
        "Second",
    )


def test_multi_signature_ignores_unrelated_active_mesh():
    first = _mesh("First", 401)
    second = _mesh("Second", 402)
    context_a = SimpleNamespace(
        scene=_scene(),
        selected_objects=(second, first),
        active_object=_mesh("UnrelatedA", 403),
    )
    context_b = SimpleNamespace(
        scene=context_a.scene,
        selected_objects=(second, first),
        active_object=_mesh("UnrelatedB", 404),
    )

    signature_a = a1_export_readiness.build_a1_readiness_signature(context_a)
    signature_b = a1_export_readiness.build_a1_readiness_signature(context_b)

    assert signature_a == signature_b
    assert a1_export_readiness._requested_object_ids(context_a) == (
        "First",
        "Second",
    )


def test_cached_report_becomes_stale_when_request_signature_changes(monkeypatch):
    scene = SimpleNamespace(as_pointer=lambda: 501)
    context = SimpleNamespace(scene=scene)
    report = _report(signature="first")
    monkeypatch.setattr(
        a1_export_readiness,
        "build_a1_readiness_signature",
        lambda _context: "first",
    )
    a1_export_readiness.clear_a1_export_readiness()
    try:
        a1_export_readiness.store_a1_export_readiness(context, report)
        state, current = a1_export_readiness.current_a1_export_readiness(context)
        assert state is A1ReadinessState.READY
        assert current is report

        monkeypatch.setattr(
            a1_export_readiness,
            "build_a1_readiness_signature",
            lambda _context: "changed",
        )
        state, current = a1_export_readiness.current_a1_export_readiness(context)
        assert state is A1ReadinessState.STALE
        assert current is report
    finally:
        a1_export_readiness.clear_a1_export_readiness()


def test_depsgraph_update_invalidates_cached_report(monkeypatch):
    scene = SimpleNamespace(as_pointer=lambda: 602)
    context = SimpleNamespace(scene=scene)
    report = _report(signature="same")
    monkeypatch.setattr(
        a1_export_readiness,
        "build_a1_readiness_signature",
        lambda _context: "same",
    )
    a1_export_readiness.clear_a1_export_readiness()
    try:
        a1_export_readiness.store_a1_export_readiness(context, report)
        depsgraph = SimpleNamespace(
            updates=(SimpleNamespace(id=SimpleNamespace(id_type="MESH")),)
        )
        a1_export_readiness.a1_readiness_depsgraph_update_post(scene, depsgraph)

        state, current = a1_export_readiness.current_a1_export_readiness(context)
        assert state is A1ReadinessState.STALE
        assert current is report
    finally:
        a1_export_readiness.clear_a1_export_readiness()


def test_irrelevant_depsgraph_update_keeps_cached_report_current(monkeypatch):
    scene = SimpleNamespace(as_pointer=lambda: 703)
    context = SimpleNamespace(scene=scene)
    report = _report(signature="same")
    monkeypatch.setattr(
        a1_export_readiness,
        "build_a1_readiness_signature",
        lambda _context: "same",
    )
    a1_export_readiness.clear_a1_export_readiness()
    try:
        a1_export_readiness.store_a1_export_readiness(context, report)
        depsgraph = SimpleNamespace(
            updates=(SimpleNamespace(id=SimpleNamespace(id_type="SOUND")),)
        )
        a1_export_readiness.a1_readiness_depsgraph_update_post(scene, depsgraph)

        state, current = a1_export_readiness.current_a1_export_readiness(context)
        assert state is A1ReadinessState.READY
        assert current is report
    finally:
        a1_export_readiness.clear_a1_export_readiness()


def test_export_guard_requires_current_non_blocked_report(monkeypatch):
    scene = SimpleNamespace(as_pointer=lambda: 804)
    context = SimpleNamespace(scene=scene)
    monkeypatch.setattr(
        a1_export_readiness,
        "build_a1_readiness_signature",
        lambda _context: "blocked",
    )
    blocked = _report(
        signature="blocked",
        global_issues=(_issue(IssueSeverity.ERROR, code="BLOCK"),),
    )
    a1_export_readiness.clear_a1_export_readiness()
    try:
        allowed, message = a1_export_readiness.require_current_a1_export_readiness(
            context
        )
        assert not allowed
        assert "Analyze" in message

        a1_export_readiness.store_a1_export_readiness(context, blocked)
        allowed, message = a1_export_readiness.require_current_a1_export_readiness(
            context
        )
        assert not allowed
        assert "blocked by 1" in message
    finally:
        a1_export_readiness.clear_a1_export_readiness()
