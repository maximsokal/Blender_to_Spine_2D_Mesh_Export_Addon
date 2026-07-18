from types import SimpleNamespace

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MultiObjectMode,
    ExportResult,
    IssueSeverity,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import a1_ui_router
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_ui_rna import (
    _ObjectExportProfile,
)


def test_single_connect_fallback_is_visible_without_debug_logging(tmp_path, monkeypatch):
    first_object = object()
    second_object = object()
    profiles = (
        _ObjectExportProfile(
            source_object=first_object,
            object_name="Hero",
            sequence_start_frame=0,
            sequence_frame_count=0,
            connect_enabled=True,
        ),
        _ObjectExportProfile(
            source_object=second_object,
            object_name="Weapon",
            sequence_start_frame=0,
            sequence_frame_count=0,
            connect_enabled=False,
        ),
    )
    sources = (
        SimpleNamespace(component_id="object_1:Hero"),
        SimpleNamespace(component_id="object_2:Weapon"),
    )
    captured = {}

    monkeypatch.setattr(
        a1_ui_router,
        "_ordered_selected_meshes",
        lambda _context: (first_object, second_object),
    )
    monkeypatch.setattr(
        a1_ui_router,
        "_capture_scene_profile",
        lambda _scene: SimpleNamespace(output_directory=tmp_path),
    )
    monkeypatch.setattr(
        a1_ui_router,
        "_capture_selected_profiles",
        lambda _objects: profiles,
    )
    monkeypatch.setattr(
        a1_ui_router,
        "_build_sources_from_profiles",
        lambda _profiles, _scene: sources,
    )

    def export_multi(resolved_sources, settings, **_kwargs):
        captured["sources"] = resolved_sources
        captured["settings"] = settings
        return ExportResult(
            success=True,
            statistics={"route": settings.mode.value},
        )

    monkeypatch.setattr(a1_ui_router, "export_a1_multi_object", export_multi)
    context = SimpleNamespace(scene=object())

    result = a1_ui_router.export_selected_objects_a1(context)

    assert captured["sources"] == sources
    assert captured["settings"].mode is A1MultiObjectMode.STANDALONE
    assert result.success is True
    assert result.statistics["single_connect_fallback_count"] == 1
    assert result.statistics["route"] == A1MultiObjectMode.STANDALONE.value
    issue = result.issues[0]
    assert issue.severity is IssueSeverity.WARNING
    assert issue.code == "A1_SINGLE_CONNECT_FALLBACK"
    assert issue.object_id == "Hero"
    assert issue.context == {
        "selected_object_count": 2,
        "connected_object_count": 1,
        "fallback_mode": A1MultiObjectMode.STANDALONE.value,
    }
