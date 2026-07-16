from Blender_to_Spine2D_Mesh_Exporter.domain.baking import BakeMode


def test_executable_bake_modes_match_supported_blender_44_pipeline():
    assert tuple(item.value for item in BakeMode) == (
        "DIFFUSE",
        "COMBINED",
        "EMIT",
    )
