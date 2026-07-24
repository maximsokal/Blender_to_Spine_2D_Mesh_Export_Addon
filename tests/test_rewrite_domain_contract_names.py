from dataclasses import fields
from pathlib import Path

import pytest

import Blender_to_Spine2D_Mesh_Exporter.application as application
from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1MultiObjectExportSettings,
    ConnectedCameraRenderPolicy,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakeFrameTask,
    BakeMaterialPolicy,
    BakeMode,
    BakePlan,
    BakeSettings,
    BakeStrategyId,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import A1AngularMode


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
ACTIVE_DIRECTORIES = (
    PACKAGE / "application",
    PACKAGE / "blender_adapter",
    PACKAGE / "domain",
    PACKAGE / "infrastructure",
)
ACTIVE_ROOT_FILES = (
    PACKAGE / "__init__.py",
    PACKAGE / "addon_preferences.py",
    PACKAGE / "config.py",
    PACKAGE / "single_object_operator.py",
    PACKAGE / "ui.py",
)
RETIRED_IDENTIFIERS = (
    "LEGACY_SEED_CONE",
    "LEGACY_ANY_IMAGE",
    "LEGACY_SINGLE_PASS",
    "ConnectedB4RenderPolicy",
    "connected_b4_render_policy",
)
RAW_PROJECTION_IMPORT = "from .a1_attachment_projection import"
RAW_PROJECTION_IMPORT_OWNER = (
    PACKAGE / "application" / "a1_attachment_projection_service.py"
)


def _active_sources():
    paths = list(ACTIVE_ROOT_FILES)
    for directory in ACTIVE_DIRECTORIES:
        paths.extend(directory.rglob("*.py"))
    return tuple(sorted(set(paths), key=lambda path: path.as_posix().casefold()))


def test_active_rewrite_contains_no_retired_domain_identifiers():
    findings = []
    for path in _active_sources():
        source = path.read_text(encoding="utf-8")
        for token in RETIRED_IDENTIFIERS:
            if token in source:
                findings.append(f"{path.relative_to(ROOT)}: {token}")
    assert findings == []


def test_retired_serialized_enum_values_are_rejected():
    with pytest.raises(ValueError):
        A1AngularMode("LEGACY_SEED_CONE")
    with pytest.raises(ValueError):
        BakeMaterialPolicy("LEGACY_ANY_IMAGE")
    assert not hasattr(BakeStrategyId, "LEGACY_SINGLE_PASS")


def test_connected_camera_policy_has_one_current_contract():
    assert application.ConnectedCameraRenderPolicy is ConnectedCameraRenderPolicy
    assert not hasattr(application, "ConnectedB4RenderPolicy")
    names = {field.name for field in fields(A1MultiObjectExportSettings)}
    assert "connected_camera_render_policy" in names
    assert "connected_b4_render_policy" not in names


def test_production_uses_one_normalized_attachment_projection_entry_point():
    owners = []
    for path in (PACKAGE / "application").rglob("*.py"):
        if RAW_PROJECTION_IMPORT in path.read_text(encoding="utf-8"):
            owners.append(path)

    assert owners == [RAW_PROJECTION_IMPORT_OWNER]
    public_source = (PACKAGE / "application" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assembly_source = (
        PACKAGE / "application" / "a1_document_assembly.py"
    ).read_text(encoding="utf-8")
    camera_source = (
        PACKAGE / "application" / "a1_camera_projection.py"
    ).read_text(encoding="utf-8")
    assert "a1_attachment_projection_service" in public_source
    assert "a1_attachment_projection_service" in assembly_source
    assert "a1_attachment_projection_service" in camera_source


def test_bake_plan_requires_explicit_strategy_passes(tmp_path):
    settings = BakeSettings(
        width=64,
        height=64,
        output_directory=tmp_path,
        output_stem="Object",
    )
    analysis = ObjectMaterialAnalysis(
        "Object",
        (
            MaterialAnalysis(
                slot_index=0,
                material_name="Material",
                kind=MaterialKind.SOLID_COLOR,
                node_types=("BSDF_PRINCIPLED", "OUTPUT_MATERIAL"),
            ),
        ),
    )
    frame_task = BakeFrameTask(
        task_index=0,
        timeline_frame=None,
        image_name="Object_Baked",
        output_path=tmp_path / "Object_Baked.png",
    )

    with pytest.raises(ValueError, match="passes must be a non-empty tuple"):
        BakePlan(
            source_object_id="Object",
            settings=settings,
            material_analysis=analysis,
            bake_mode=BakeMode.DIFFUSE,
            frame_tasks=(frame_task,),
        )
