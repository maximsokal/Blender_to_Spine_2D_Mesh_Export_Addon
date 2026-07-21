from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "validator.py"
)
SERIALIZER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "serializer.py"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_raw_and_typed_attachments_share_one_mesh_payload_validator():
    source = read(VALIDATOR)

    raw_index = source.index("def _validate_raw_attachment(")
    typed_index = source.index("def _validate_mesh_attachment(")
    payload_index = source.index("def _validate_mesh_payload(")

    assert source.count("self._validate_mesh_payload(") == 2
    assert raw_index < payload_index
    assert typed_index < payload_index


def test_topology_runs_only_after_shape_and_index_validation():
    source = read(VALIDATOR)

    triangle_index_check = source.index("triangle_index_issues =")
    triangle_topology = source.index("self._validate_triangle_topology(")
    edge_index_check = source.index("edge_index_issues =")
    edge_topology = source.index("self._validate_edge_topology(")

    assert triangle_index_check < triangle_topology
    assert "and not triangle_index_issues" in source
    assert edge_index_check < edge_topology
    assert "and not edge_index_issues" in source


def test_validator_owns_explicit_triangle_and_edge_issue_codes():
    source = read(VALIDATOR)

    for code in (
        "EMPTY_TRIANGLE_ARRAY",
        "DEGENERATE_TRIANGLE",
        "DUPLICATE_TRIANGLE",
        "UNUSED_MESH_VERTEX",
        "SELF_EDGE",
        "DUPLICATE_EDGE",
    ):
        assert f'"{code}"' in source

    assert "def _validate_triangle_topology(" in source
    assert "def _validate_edge_topology(" in source
    assert "tuple(sorted(triangle))" in source
    assert "if first == second:" in source


def test_serializer_cannot_bypass_complete_spine_validation():
    source = read(SERIALIZER)

    to_dict_index = source.index("def to_dict(")
    validation_index = source.index(
        "self._validator.validate_or_raise(document)",
        to_dict_index,
    )
    serialization_index = source.index(
        "data: dict[str, Any] = {",
        validation_index,
    )

    assert to_dict_index < validation_index < serialization_index
