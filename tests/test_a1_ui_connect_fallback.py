from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MultiObjectMode,
    ExportResult,
    IssueSeverity,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import a1_ui_router


def test_single_connect_fallback_is_visible_without_debug_logging(tmp_path, monkeypatch):
    first_object = object()
    second_object = object()
    sources = (
        SimpleNamespace(component_id="object_1:Hero"),
        SimpleNamespace(component_id="object_2:Weapon"),
    )
    issue = a1_ui_router.ExportIssue(
        severity=IssueSeverity.WARNING,
        stage="VALIDATE_REQUEST",
        code="A1_SINGLE_CONNECT_FALLBACK",
        message="Exactly one selected object has Connect enabled.",
        object_id="Hero",
        context={
            "selected_object_count": 2,
            "connected_object_count": 1,
            "fallback_mode": A1MultiObjectMode.STANDALONE.value,
        },
    )
    settings = SimpleNamespace(mode=A1MultiObjectMode.STANDALONE)
    plan = SimpleNamespace(
        settings=settings,
        all_sources=sources,
        connected_sources=(),
        standalone_sources=sources,
        issues=(issue,),
    )
    monkeypatch.setattr(a1_ui_router, "build_selected_ui_export_plan", lambda _context: plan)

    captured = {}
    def export_multi(resolved_sources, resolved_settings, **_kwargs):
        captured["sources"] = resolved_sources
        captured["settings"] = resolved_settings
        return ExportResult(success=True, statistics={"route": resolved_settings.mode.value})

    monkeypatch.setattr(a1_ui_router, "export_a1_multi_object", export_multi)
    context = SimpleNamespace(scene=object())
    result = a1_ui_router.export_selected_objects_a1(
        context,
        progress_callback=lambda _update: None,
    )

    assert captured["sources"] == sources
    assert captured["settings"].mode is A1MultiObjectMode.STANDALONE
    assert result.success is True
    assert result.statistics["ui_request_warning_count"] == 1
    assert result.statistics["route"] == A1MultiObjectMode.STANDALONE.value
    assert result.issues == (issue,)
