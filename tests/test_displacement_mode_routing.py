"""Regression coverage for live displacement-mode Normal/UV routing."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_displacement import (
    DISPLACEMENT_BUMP_CONTEXT_CODE,
    DISPLACEMENT_RENDER_REQUIRED_CODE,
    apply_displacement_method_boundary,
    material_displacement_method,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.production_shader_capability_routing import (
    _normal_uv_allows_bump_displacement,
    _normal_uv_blocking_camera_findings,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    BakePlanError,
    MaterialAnalysis,
    MaterialCapabilityAudit,
    MaterialGraphSnapshot,
    MaterialKind,
    MaterialSemanticChannel,
    ObjectMaterialAnalysis,
    ShaderBakeCapability,
    ShaderCapabilityFinding,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking.normal_uv_camera_context import (
    _validate_normal_uv_channels,
)


@dataclass
class _Material:
    displacement_method: str | None = None
    cycles: object | None = None


def _displacement_audit() -> MaterialCapabilityAudit:
    return MaterialCapabilityAudit(
        material_name="1006",
        render_target="CYCLES",
        required_capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
        findings=(
            ShaderCapabilityFinding(
                capability=ShaderBakeCapability.CAMERA_RENDER_REQUIRED,
                code=DISPLACEMENT_RENDER_REQUIRED_CODE,
                reason="Render-evaluated displacement requires camera projection",
            ),
        ),
    )


def _analysis_with_channels(
    *channels: MaterialSemanticChannel,
) -> ObjectMaterialAnalysis:
    graph = MaterialGraphSnapshot(
        material_name="1006",
        active_output_node_id=None,
        reachable_nodes=(),
        reachable_links=(),
        semantic_channels=tuple(channels),
        dependencies=(),
    )
    return ObjectMaterialAnalysis(
        source_object_id="Cube",
        slots=(
            MaterialAnalysis(
                slot_index=0,
                material_name="1006",
                kind=MaterialKind.PROCEDURAL,
                graph=graph,
            ),
        ),
    )


def test_bump_only_displacement_becomes_camera_combined_safe() -> None:
    audit = apply_displacement_method_boundary(
        _displacement_audit(),
        _Material(displacement_method="BUMP"),
    )

    assert tuple(finding.code for finding in audit.findings) == (
        DISPLACEMENT_BUMP_CONTEXT_CODE,
    )
    assert audit.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
    assert _normal_uv_blocking_camera_findings((audit,)) == ()
    assert _normal_uv_allows_bump_displacement((audit,))


def test_true_displacement_stays_blocked() -> None:
    audit = apply_displacement_method_boundary(
        _displacement_audit(),
        _Material(displacement_method="DISPLACEMENT"),
    )

    assert audit == _displacement_audit()
    assert not _normal_uv_allows_bump_displacement((audit,))
    assert _normal_uv_blocking_camera_findings((audit,)) == (
        (
            "1006",
            ((DISPLACEMENT_RENDER_REQUIRED_CODE, None, None),),
        ),
    )


def test_displacement_and_bump_stays_blocked() -> None:
    audit = apply_displacement_method_boundary(
        _displacement_audit(),
        _Material(displacement_method="BOTH"),
    )

    assert audit == _displacement_audit()
    assert not _normal_uv_allows_bump_displacement((audit,))
    assert _normal_uv_blocking_camera_findings((audit,))


def test_unknown_displacement_rna_stays_fail_closed() -> None:
    audit = apply_displacement_method_boundary(
        _displacement_audit(),
        _Material(displacement_method="SOMETHING_NEW"),
    )

    assert audit == _displacement_audit()
    assert not _normal_uv_allows_bump_displacement((audit,))
    assert _normal_uv_blocking_camera_findings((audit,))


def test_legacy_cycles_displacement_method_is_supported_without_guessing() -> None:
    material = _Material(
        displacement_method=None,
        cycles=SimpleNamespace(displacement_method="BUMP"),
    )

    assert material_displacement_method(material) == "BUMP"
    audit = apply_displacement_method_boundary(_displacement_audit(), material)
    assert _normal_uv_blocking_camera_findings((audit,)) == ()
    assert _normal_uv_allows_bump_displacement((audit,))


def test_audit_without_displacement_finding_is_unchanged() -> None:
    audit = MaterialCapabilityAudit(
        material_name="Plain",
        render_target="CYCLES",
        required_capability=ShaderBakeCapability.LOCAL_UV_SAFE,
        findings=(
            ShaderCapabilityFinding(
                capability=ShaderBakeCapability.LOCAL_UV_SAFE,
                code="LOCAL_GRAPH",
                reason="local",
            ),
        ),
    )

    assert apply_displacement_method_boundary(
        audit,
        _Material(displacement_method="BUMP"),
    ) is audit
    assert not _normal_uv_allows_bump_displacement((audit,))


def test_domain_normal_uv_rejects_displacement_without_live_bump_authorization() -> None:
    analysis = _analysis_with_channels(
        MaterialSemanticChannel.SURFACE_COLOR,
        MaterialSemanticChannel.DISPLACEMENT,
    )

    with pytest.raises(BakePlanError, match="true render displacement"):
        _validate_normal_uv_channels(
            analysis,
            allow_bump_displacement=False,
        )


def test_domain_normal_uv_accepts_displacement_after_live_bump_authorization() -> None:
    analysis = _analysis_with_channels(
        MaterialSemanticChannel.SURFACE_COLOR,
        MaterialSemanticChannel.DISPLACEMENT,
    )

    _validate_normal_uv_channels(
        analysis,
        allow_bump_displacement=True,
    )


def test_domain_normal_uv_still_rejects_volume_when_bump_is_authorized() -> None:
    analysis = _analysis_with_channels(
        MaterialSemanticChannel.SURFACE_COLOR,
        MaterialSemanticChannel.DISPLACEMENT,
        MaterialSemanticChannel.VOLUME,
    )

    with pytest.raises(BakePlanError, match="VOLUME"):
        _validate_normal_uv_channels(
            analysis,
            allow_bump_displacement=True,
        )
