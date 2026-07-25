from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "blender_adapter"


def test_rewrite_material_bake_has_no_legacy_diffuse_color_fallback():
    source = (ADAPTER / "bake_materials.py").read_text(encoding="utf-8")

    assert "def _material_diffuse_rgba" not in source
    assert "source_material.use_nodes" not in source
    assert "source_material.diffuse_color" not in source


def test_generated_materials_mutate_only_temporary_copies():
    source = (ADAPTER / "bake_materials.py").read_text(encoding="utf-8")

    assert "material.diffuse_color" in source
    assert "source_material.diffuse_color =" not in source
