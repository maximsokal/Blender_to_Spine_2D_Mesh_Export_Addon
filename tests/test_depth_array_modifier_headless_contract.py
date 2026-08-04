"""Static contract for the Blender Array-modifier Depth regression runner."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "blender_headless" / "run_depth_array_modifier_integration.py"
IDENTITY = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "geometry"
    / "evaluated_identity.py"
)
DEPTH_PREPARATION = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "blender_adapter"
    / "a1_depth_source_geometry_preparation.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_array_runner_uses_real_modifier_and_public_depth_preparation() -> None:
    source = _read(RUNNER)

    assert 'type="ARRAY"' in source
    assert "modifier.count = _COPY_COUNT" in source
    assert "modifier.use_merge_vertices = False" in source
    assert "read_evaluated_mesh_snapshot(" in source
    assert "ModifierLineagePolicy.ALLOW_SOURCE_DUPLICATION" in source
    assert "rebase_mesh_snapshot_to_evaluated_identity(raw.snapshot)" in source
    assert "prepare_a1_object(" in source
    assert "PreparedDepthA1Object" in source
    assert 'issue.code == "EVALUATED_IDENTITY_REBASED"' in source
    assert "_object_fingerprint(source, modifier) == object_before" in source
    assert "_temporary_datablock_names() == temporary_before" in source
    assert "[DEPTH-ARRAY-MODIFIER] PASS" in source


def test_evaluated_identity_rebases_every_topology_domain_without_bpy() -> None:
    source = _read(IDENTITY)

    assert "def rebase_mesh_snapshot_to_evaluated_identity(" in source
    assert "SourceVertexId(object_id, vertex.id.index)" in source
    assert "SourceEdgeId(object_id, edge.id.index)" in source
    assert "SourceFaceId(object_id, face.id.index)" in source
    assert "def _loop_local_identity(" in source
    assert "MeshSnapshotValidator().validate_or_raise" not in source
    assert "validator.validate_or_raise(snapshot)" in source
    assert "validator.validate_or_raise(resolved_snapshot)" in source
    assert "import bpy" not in source
    assert "import bmesh" not in source
    assert "bpy.ops" not in source


def test_depth_route_validates_provenance_before_local_identity_rebase() -> None:
    source = _read(DEPTH_PREPARATION)
    read_index = source.index("source_snapshot, modifier_count, warnings, uv_report")
    rebase_index = source.index("_canonicalize_depth_evaluated_identity(")
    normalize_index = source.index("normalized = _normalize_source_geometry(")

    assert "modifier_lineage_policy=ModifierLineagePolicy.ALLOW_SOURCE_DUPLICATION" in source
    assert "def _canonicalize_depth_evaluated_identity(" in source
    assert "rebase_mesh_snapshot_to_evaluated_identity(snapshot)" in source
    assert 'code="EVALUATED_IDENTITY_REBASED"' in source
    assert read_index < rebase_index < normalize_index
