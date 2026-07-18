from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.infrastructure.pipeline_static_audit import (
    AuditSeverity,
    audit_module_source,
    audit_pipeline_package,
)


def _codes(audit):
    return {item.code for item in audit.findings}


def test_domain_blender_import_and_layer_violation_are_errors():
    audit = audit_module_source(
        "import bpy\nfrom ..blender_adapter import mesh_reader\n",
        module="domain.bad",
        relative_path="domain/bad.py",
        layer="domain",
        package_name="Blender_to_Spine2D_Mesh_Exporter",
    )
    assert "BLENDER_IMPORT_OUTSIDE_ADAPTER" in _codes(audit)
    assert "LAYER_IMPORT_VIOLATION" in _codes(audit)
    assert all(
        item.severity is AuditSeverity.ERROR
        for item in audit.findings
        if item.code in {"BLENDER_IMPORT_OUTSIDE_ADAPTER", "LAYER_IMPORT_VIOLATION"}
    )


def test_bmesh_lifetime_and_bpy_ops_loop_are_reported():
    source = """
import bmesh
import bpy

def broken():
    bm = bmesh.new()
    for _ in range(2):
        bpy.ops.object.mode_set(mode='OBJECT')
"""
    audit = audit_module_source(
        source,
        module="blender_adapter.bad",
        relative_path="blender_adapter/bad.py",
        layer="blender_adapter",
        package_name="Blender_to_Spine2D_Mesh_Exporter",
    )
    assert "BMESH_FREE_MISSING" in _codes(audit)
    assert "BPY_OPS_IN_LOOP" in _codes(audit)


def test_bmesh_free_in_finally_is_accepted():
    source = """
import bmesh

def safe():
    bm = bmesh.new()
    try:
        return len(bm.verts)
    finally:
        bm.free()
"""
    audit = audit_module_source(
        source,
        module="blender_adapter.safe",
        relative_path="blender_adapter/safe.py",
        layer="blender_adapter",
        package_name="Blender_to_Spine2D_Mesh_Exporter",
    )
    assert "BMESH_FREE_MISSING" not in _codes(audit)
    assert "BMESH_FREE_NOT_GUARANTEED" not in _codes(audit)


def test_package_audit_scans_every_production_file(tmp_path: Path):
    package = tmp_path / "addon"
    (package / "domain").mkdir(parents=True)
    (package / "application").mkdir()
    (package / "blender_adapter").mkdir()
    (package / "infrastructure").mkdir()
    (package / "domain" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package / "application" / "service.py").write_text(
        "from ..domain import model\n", encoding="utf-8"
    )
    (package / "blender_adapter" / "adapter.py").write_text(
        "from ..application import service\n", encoding="utf-8"
    )
    (package / "infrastructure" / "files.py").write_text(
        "from pathlib import Path\n", encoding="utf-8"
    )
    report = audit_pipeline_package(package, package_name="addon")
    assert report["summary"]["module_count"] == 4
    assert report["summary"]["error_count"] == 0


def test_package_init_relative_import_stays_in_its_layer():
    source = "from .a1_single_object import A1SingleObjectStage\n"
    audit = audit_module_source(
        source,
        module="application",
        relative_path="application/__init__.py",
        layer="application",
        package_name="Blender_to_Spine2D_Mesh_Exporter",
    )
    assert audit.internal_imports == ("application.a1_single_object",)
    assert not [item for item in audit.findings if item.code == "LAYER_IMPORT_VIOLATION"]
