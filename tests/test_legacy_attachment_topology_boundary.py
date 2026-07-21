from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ATTACHMENT = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "legacy_attachment_builder.py"
)


def read_source() -> str:
    return LEGACY_ATTACHMENT.read_text(encoding="utf-8")


def test_request_rejects_empty_and_degenerate_triangle_topology():
    source = read_source()

    assert "triangles must contain at least one triangle" in source
    assert "is degenerate" in source
    assert "triangles contain duplicate geometry" in source
    assert "every attachment vertex must be referenced by a triangle" in source


def test_request_rejects_invalid_edge_topology():
    source = read_source()

    assert "is a self-edge" in source
    assert "edges contain duplicate undirected pair" in source


def test_topology_validation_runs_after_strict_index_validation():
    source = read_source()

    index_validation = source.index('f"{field_name}[{value_index}]"')
    triangle_validation = source.index("triangle_keys: set[tuple[int, int, int]]")
    edge_validation = source.index("edge_keys: set[tuple[int, int]]")

    assert index_validation < triangle_validation < edge_validation
