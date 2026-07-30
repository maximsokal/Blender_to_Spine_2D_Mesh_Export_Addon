from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    A1SingleObjectStage,
    ExportIssue,
    ExportSettings,
    IssueSeverity,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import a1_object_preparation
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_preparation_contracts import (
    A1ObjectPreparationError,
)


def _settings() -> A1SingleObjectExportSettings:
    return A1SingleObjectExportSettings(
        export=ExportSettings(
            texture_width=64,
            texture_height=64,
            output_directory=Path.cwd(),
        )
    )


def _warning(code: str) -> ExportIssue:
    return ExportIssue(
        severity=IssueSeverity.WARNING,
        stage=A1SingleObjectStage.VALIDATE_REQUEST.value,
        code=code,
        message=code,
        object_id="Hero",
    )


def _stage_values():
    source = SimpleNamespace(
        source_object=object(),
        object_id="Hero",
        prefix="Hero",
        settings=object(),
        output_paths=object(),
        source_snapshot=object(),
        z_groups=object(),
        geometry=object(),
        warnings=(_warning("source-warning"),),
        statistics={"source_vertices": 4},
    )
    uv = SimpleNamespace(
        texturing_topology=object(),
        unwrap_result=object(),
        uv_regions=object(),
        warnings=(_warning("uv-warning"),),
        statistics={"uv_loop_count": 6},
    )
    texture = SimpleNamespace(
        material_analysis=object(),
        bake_plan=object(),
        warnings=(_warning("texture-warning"),),
        statistics={"bake_pass_count": 1},
    )
    document = SimpleNamespace(
        rig=object(),
        document_assembly=object(),
        warnings=(_warning("document-warning"),),
        statistics={"final_bone_count": 3},
    )
    return source, uv, texture, document


def test_orchestrator_passes_each_typed_stage_result_to_the_next(monkeypatch):
    source, uv, texture, document = _stage_values()
    calls = []
    progress = []
    captured = {}

    def source_stage(source_obj, settings, *, scene):
        calls.append(("source", source_obj, settings, scene))
        return source

    def uv_stage(value, *, context, scene):
        assert value is source
        calls.append(("uv", context, scene))
        return uv

    def texture_stage(value, *, context, scene):
        assert value is uv
        calls.append(("texture", context, scene))
        return texture

    def document_stage(value):
        assert value is texture
        calls.append(("document",))
        return document

    def prepared_builder(**kwargs):
        captured.update(kwargs)
        return "prepared"

    monkeypatch.setattr(a1_object_preparation, "prepare_a1_source_geometry", source_stage)
    monkeypatch.setattr(a1_object_preparation, "prepare_a1_uv", uv_stage)
    monkeypatch.setattr(a1_object_preparation, "prepare_a1_texture_plan", texture_stage)
    monkeypatch.setattr(a1_object_preparation, "prepare_a1_document", document_stage)
    monkeypatch.setattr(a1_object_preparation, "PreparedA1Object", prepared_builder)

    source_obj = object()
    settings = _settings()
    context = object()
    scene = object()
    result = a1_object_preparation.prepare_a1_object(
        source_obj,
        settings,
        context=context,
        scene=scene,
        progress_callback=progress.append,
    )

    assert result == "prepared"
    assert [item[0] for item in calls] == ["source", "uv", "texture", "document"]
    assert [item.percent for item in progress] == [0, 10, 45, 65, 82, 100]
    assert [item.stage for item in progress] == [
        A1SingleObjectStage.VALIDATE_REQUEST.value,
        A1SingleObjectStage.READ_GEOMETRY.value,
        A1SingleObjectStage.BUILD_TEXTURING_TOPOLOGY.value,
        A1SingleObjectStage.ANALYZE_MATERIALS.value,
        A1SingleObjectStage.BUILD_RIG.value,
        A1SingleObjectStage.ASSEMBLE_DOCUMENT.value,
    ]
    assert captured["object_id"] == "Hero"
    assert captured["warnings"] == document.warnings
    assert captured["statistics"] == document.statistics


def test_typed_stage_error_is_not_rewrapped(monkeypatch):
    error = A1ObjectPreparationError(
        stage=A1SingleObjectStage.READ_GEOMETRY,
        object_id="Hero",
        cause=RuntimeError("mesh failed"),
        statistics={"source_vertices": 0},
        warnings=(),
    )

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(a1_object_preparation, "prepare_a1_source_geometry", fail)

    with pytest.raises(A1ObjectPreparationError) as captured:
        a1_object_preparation.prepare_a1_object(object(), _settings())
    assert captured.value is error


def test_unexpected_stage_error_keeps_current_stage_and_partial_diagnostics(monkeypatch):
    source, _uv, _texture, _document = _stage_values()
    monkeypatch.setattr(
        a1_object_preparation,
        "prepare_a1_source_geometry",
        lambda *_args, **_kwargs: source,
    )

    def fail_uv(*_args, **_kwargs):
        raise RuntimeError("unexpected unwrap failure")

    monkeypatch.setattr(a1_object_preparation, "prepare_a1_uv", fail_uv)

    with pytest.raises(A1ObjectPreparationError) as captured:
        a1_object_preparation.prepare_a1_object(object(), _settings())

    error = captured.value
    assert error.stage is A1SingleObjectStage.BUILD_TEXTURING_TOPOLOGY
    assert error.object_id == "Hero"
    assert dict(error.statistics) == source.statistics
    assert error.warnings == source.warnings
