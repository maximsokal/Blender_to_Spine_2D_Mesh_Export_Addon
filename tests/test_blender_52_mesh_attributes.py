"""Blender-independent regressions for Blender 5.2 Mesh attribute adapters."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_edge_attributes import (
    MeshEdgeAttributeError,
    SHARP_EDGE_ATTRIBUTE,
    UV_SEAM_ATTRIBUTE,
    read_boolean_edge_attribute,
    write_boolean_edge_attribute,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.mesh_uv_attributes import (
    MeshUvAttributeError,
    read_uv_coordinate,
    read_uv_coordinates,
    write_uv_coordinate,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "Blender_to_Spine2D_Mesh_Exporter"
ADAPTER = PACKAGE / "blender_adapter"


class _BooleanAttribute:
    def __init__(self, name: str, domain: str, data_type: str, length: int):
        self.name = name
        self.domain = domain
        self.data_type = data_type
        self.data = [SimpleNamespace(value=False) for _ in range(length)]


class _AttributeCollection(dict):
    def new(self, *, name: str, type: str, domain: str):
        attribute = _BooleanAttribute(name, domain, type, self.edge_count)
        self[name] = attribute
        return attribute


class _Mesh:
    def __init__(self, edge_count: int):
        self.edges = tuple(object() for _ in range(edge_count))
        self.attributes = _AttributeCollection()
        self.attributes.edge_count = edge_count


class _UvValue:
    def __init__(self, vector):
        self.vector = tuple(vector)


class _UvLayer:
    def __init__(self, name: str, coordinates):
        self.name = name
        self.uv = [_UvValue(value) for value in coordinates]


def test_missing_blender_edge_attribute_reads_as_false():
    mesh = _Mesh(3)

    assert read_boolean_edge_attribute(mesh, UV_SEAM_ATTRIBUTE) == (
        False,
        False,
        False,
    )


def test_boolean_edge_attribute_round_trip_uses_edge_domain():
    mesh = _Mesh(3)

    attribute = write_boolean_edge_attribute(
        mesh,
        SHARP_EDGE_ATTRIBUTE,
        (True, False, True),
    )

    assert attribute is mesh.attributes[SHARP_EDGE_ATTRIBUTE]
    assert attribute.domain == "EDGE"
    assert attribute.data_type == "BOOLEAN"
    assert read_boolean_edge_attribute(mesh, SHARP_EDGE_ATTRIBUTE) == (
        True,
        False,
        True,
    )


def test_all_false_edge_attribute_can_be_omitted():
    mesh = _Mesh(2)

    result = write_boolean_edge_attribute(
        mesh,
        UV_SEAM_ATTRIBUTE,
        (False, False),
        omit_when_all_false=True,
    )

    assert result is None
    assert UV_SEAM_ATTRIBUTE not in mesh.attributes


def test_existing_edge_attribute_must_have_boolean_edge_contract():
    mesh = _Mesh(2)
    mesh.attributes[UV_SEAM_ATTRIBUTE] = _BooleanAttribute(
        UV_SEAM_ATTRIBUTE,
        "POINT",
        "BOOLEAN",
        2,
    )

    with pytest.raises(MeshEdgeAttributeError, match="EDGE domain"):
        read_boolean_edge_attribute(mesh, UV_SEAM_ATTRIBUTE)


def test_edge_attribute_writer_rejects_non_boolean_values():
    mesh = _Mesh(2)

    with pytest.raises(TypeError, match="bool values only"):
        write_boolean_edge_attribute(mesh, UV_SEAM_ATTRIBUTE, (True, 1))


def test_uv_attribute_read_write_round_trip_uses_vector_collection():
    layer = _UvLayer("BakeUV", ((0.1, 0.2), (0.3, 0.4)))

    assert read_uv_coordinate(layer, 0, expected_length=2) == (0.1, 0.2)
    assert read_uv_coordinates(layer, expected_length=2) == (
        (0.1, 0.2),
        (0.3, 0.4),
    )

    write_uv_coordinate(
        layer,
        1,
        (0.75, 0.25),
        expected_length=2,
    )

    assert layer.uv[1].vector == (0.75, 0.25)


def test_uv_attribute_rejects_wrong_length_and_non_finite_values():
    layer = _UvLayer("BakeUV", ((0.1, 0.2),))

    with pytest.raises(MeshUvAttributeError, match="1 values for 2 mesh loops"):
        read_uv_coordinate(layer, 0, expected_length=2)

    with pytest.raises(MeshUvAttributeError, match="non-finite"):
        write_uv_coordinate(
            layer,
            0,
            (float("nan"), 0.0),
            expected_length=1,
        )


def test_active_rewrite_mesh_code_contains_no_retired_edge_flags():
    sources = "\n".join(
        (ADAPTER / name).read_text(encoding="utf-8")
        for name in (
            "mesh_reader.py",
            "evaluated_mesh_reader.py",
            "mesh_writer.py",
        )
    )

    assert ".use_seam" not in sources
    assert ".use_edge_sharp" not in sources
    assert 'UV_SEAM_ATTRIBUTE = "uv_seam"' in (
        ADAPTER / "mesh_edge_attributes.py"
    ).read_text(encoding="utf-8")
    assert 'SHARP_EDGE_ATTRIBUTE = "sharp_edge"' in (
        ADAPTER / "mesh_edge_attributes.py"
    ).read_text(encoding="utf-8")


def test_active_rewrite_uv_code_contains_no_mesh_uv_loop_data_access():
    for name in (
        "mesh_reader.py",
        "evaluated_mesh_reader.py",
        "mesh_writer.py",
        "uv_unwrap.py",
    ):
        source = (ADAPTER / name).read_text(encoding="utf-8")
        assert "layer.data[" not in source, name
        assert ".uv_layers.active.data" not in source, name

    uv_adapter = (ADAPTER / "mesh_uv_attributes.py").read_text(encoding="utf-8")
    assert 'getattr(layer, "uv", None)' in uv_adapter
    assert "collection[loop_index].vector" in uv_adapter


def test_evaluated_reader_keeps_to_mesh_cleanup_and_no_old_collection_fallback():
    source = (ADAPTER / "evaluated_mesh_reader.py").read_text(encoding="utf-8")
    writer = (ADAPTER / "mesh_writer.py").read_text(encoding="utf-8")

    assert "evaluated_object.to_mesh_clear()" in source
    assert "finally:" in source
    assert "remove(collection, do_unlink=True)" in source
    assert "except TypeError:" not in source
    assert "except TypeError:" not in writer


def test_uv_unwrap_always_returns_to_object_mode_in_finally():
    source = (ADAPTER / "uv_unwrap.py").read_text(encoding="utf-8")

    function = source.split("def _execute_uv_operator_plan(", 1)[1].split(
        "def unwrap_snapshot_uv(",
        1,
    )[0]
    assert "finally:" in function
    assert '_set_mode(bpy_module, "OBJECT")' in function
    assert "read_uv_coordinate(" in source
    assert "layers.active_index" not in source
