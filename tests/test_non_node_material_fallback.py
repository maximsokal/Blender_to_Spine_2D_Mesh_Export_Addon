from pathlib import Path

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.bake_materials import (
    BakeMaterialError,
    _material_diffuse_rgba,
)


class FakeMaterial:
    def __init__(self, diffuse_color):
        self.diffuse_color = diffuse_color


def test_non_node_diffuse_rgba_is_read_without_mutation():
    material = FakeMaterial((0.1, 0.7, 0.2, 1.0))

    assert _material_diffuse_rgba(material, 3) == (0.1, 0.7, 0.2, 1.0)
    assert material.diffuse_color == (0.1, 0.7, 0.2, 1.0)


def test_non_node_diffuse_rgba_rejects_invalid_values():
    material = FakeMaterial((0.1, float("nan"), 0.2, 1.0))

    try:
        _material_diffuse_rgba(material, 2)
    except BakeMaterialError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite legacy diffuse color was accepted")


def test_fallback_is_copy_only_and_transparent_legacy_material_is_rejected():
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "Blender_to_Spine2D_Mesh_Exporter"
        / "blender_adapter"
        / "bake_materials.py"
    ).read_text(encoding="utf-8")

    assert "copied.use_nodes = True" in source
    assert "node_tree.nodes.clear()" in source
    assert "source_material.use_nodes" not in source
    assert "enable material nodes so opacity can be analyzed" in source
