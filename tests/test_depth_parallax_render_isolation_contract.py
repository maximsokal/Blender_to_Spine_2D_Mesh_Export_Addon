"""Static ownership contracts for Depth parallax reserve render isolation."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
CAMERA_PLAN = PACKAGE / "domain" / "baking" / "camera_projection.py"
OBJECT_PREPARATION = PACKAGE / "blender_adapter" / "a1_object_preparation.py"
RENDER_PROXY = (
    PACKAGE / "blender_adapter" / "depth_camera_projection_render_proxy.py"
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_virtual_camera_plan_owns_exact_source_faces() -> None:
    plan = _read(CAMERA_PLAN)
    preparation = _read(OBJECT_PREPARATION)

    assert "source_face_indices: Tuple[int, ...] | None = None" in plan
    assert "non-FRONT camera projection plans require source_face_indices" in plan
    assert "FRONT plan cannot restrict source faces" in plan
    assert "source_face_indices: Tuple[int, ...]," in plan
    assert "source_face_indices=source_face_indices" in plan
    assert "source_face_indices=surface.source_face_indices" in preparation


def test_reserve_proxy_filters_only_temporary_evaluated_mesh() -> None:
    proxy = _read(RENDER_PROXY)

    assert "def _retain_virtual_view_source_faces(" in proxy
    assert "if not runtime.plan.virtual_view:" in proxy
    assert "source_face_indices = runtime.plan.source_face_indices" in proxy
    assert "mesh = bpy_module.data.meshes.new_from_object(" in proxy
    assert "_retain_virtual_view_source_faces(mesh, runtime)" in proxy
    assert "retained = set(source_face_indices)" in proxy
    assert 'bmesh.ops.delete(bm, geom=rejected, context="FACES")' in proxy
    assert "runtime.source_object.data" not in proxy


def test_reserve_proxy_bmesh_is_locally_owned_and_always_freed() -> None:
    proxy = _read(RENDER_PROXY)

    assert "bm = bmesh.new()" in proxy
    assert "bmesh.from_edit_mesh" not in proxy
    assert "finally:\n        bm.free()" in proxy
    assert proxy.count("bm.free()") == 1
