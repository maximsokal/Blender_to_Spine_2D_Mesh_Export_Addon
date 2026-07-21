from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECTION = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "application"
    / "a1_attachment_projection.py"
)


def read_source() -> str:
    return PROJECTION.read_text(encoding="utf-8")


def test_result_validates_vertex_uv_and_triangle_corner_correspondence():
    source = read_source()

    assert "hull_vertex_keys must contain at least three vertices" in source
    assert "does not match ordered key UV" in source
    assert "one entry for every triangle corner" in source
    assert "exactly match request.triangles corner order" in source


def test_projection_builds_loop_mapping_in_face_corner_order():
    source = read_source()

    loop_stream = source.index("triangle_loop_ids = tuple(")
    triangle_stream = source.index("triangles = tuple(", loop_stream)
    mapping_stream = source.index(
        "loop_to_attachment_index = tuple(zip(triangle_loop_ids, triangles))"
    )
    request_build = source.index("request = LegacyMeshAttachmentRequest(")

    assert loop_stream < triangle_stream < mapping_stream < request_build
    assert "sorted(\n            loop_keys.items()" not in source
