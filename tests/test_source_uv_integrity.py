from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.source_uv_integrity import (
    ObjectModeRequiredError,
    SourceUvIntegrityError,
    SourceUvMutationError,
    capture_source_uv_fingerprint,
    capture_source_uv_fingerprint_if_mesh,
    material_required_uv_layer_names,
    require_object_mode,
    require_source_uv_unchanged,
    require_source_uv_unchanged_if_captured,
    resolve_readable_source_uv_layer_names,
)


class _UvValue:
    def __init__(self, vector):
        self.vector = tuple(vector)


class _UvLayer:
    def __init__(self, name, coordinates, *, active_render=False):
        self.name = name
        self.uv = [_UvValue(value) for value in coordinates]
        self.active_render = active_render


class _UvLayers(list):
    def __init__(self, values, *, active=None):
        super().__init__(values)
        self.active = active


class _Mesh:
    def __init__(self, layers, *, loop_count=3, active=None):
        self.loops = [object() for _ in range(loop_count)]
        self.uv_layers = _UvLayers(layers, active=active)

    def as_pointer(self):
        return id(self)


class _NodeTree:
    def __init__(self, nodes):
        self.nodes = tuple(nodes)

    def as_pointer(self):
        return id(self)


class _AttributeNode:
    type = "ATTRIBUTE"
    node_tree = None

    def __init__(self, attribute_name):
        self.attribute_name = attribute_name


class _UvMapNode:
    type = "UVMAP"
    node_tree = None

    def __init__(self, uv_map):
        self.uv_map = uv_map


class _Object:
    type = "MESH"

    def __init__(self, mesh, nodes=()):
        self.data = mesh
        material = SimpleNamespace(node_tree=_NodeTree(nodes))
        self.material_slots = (SimpleNamespace(material=material),) if nodes else ()


class _BoundaryMode:
    def __init__(self, value):
        self.value = value


def _settings(mode="DISABLED", layer_name=None):
    return SimpleNamespace(
        source_uv_boundary_mode=_BoundaryMode(mode),
        source_uv_boundary_layer_name=layer_name,
    )


def test_malformed_unused_uv_layer_is_ignored():
    valid = _UvLayer("Valid", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    broken = _UvLayer("Broken", ())
    obj = _Object(_Mesh((valid, broken), active=valid))

    report = resolve_readable_source_uv_layer_names(obj, _settings())

    assert report.readable_layer_names == ("Valid",)
    assert report.ignored_malformed_layer_names == ("Broken",)
    assert report.malformed_required_layer_names == ()


def test_malformed_explicit_boundary_uv_layer_is_blocker():
    broken = _UvLayer("Broken", ())
    obj = _Object(_Mesh((broken,), active=broken))

    with pytest.raises(SourceUvIntegrityError, match="malformed required UV"):
        resolve_readable_source_uv_layer_names(
            obj,
            _settings("EXPLICIT_LAYER", "Broken"),
        )


def test_missing_explicit_boundary_uv_layer_is_blocker():
    obj = _Object(_Mesh((), active=None))

    with pytest.raises(SourceUvIntegrityError, match="missing required UV"):
        resolve_readable_source_uv_layer_names(
            obj,
            _settings("EXPLICIT_LAYER", "Missing"),
        )


def test_attribute_node_only_requires_real_uv_layer_names():
    uv_layer = _UvLayer("UVMap", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    obj = _Object(
        _Mesh((uv_layer,), active=uv_layer),
        nodes=(
            _AttributeNode("ColorAttribute"),
            _AttributeNode("UVMap"),
        ),
    )

    assert material_required_uv_layer_names(obj) == ("UVMap",)


def test_explicit_uv_map_node_requires_named_layer_even_when_missing():
    obj = _Object(_Mesh((), active=None), nodes=(_UvMapNode("MissingUV"),))

    with pytest.raises(SourceUvIntegrityError, match="missing required UV"):
        resolve_readable_source_uv_layer_names(obj, _settings())


def test_explicit_uv_map_node_rejects_zero_length_layer():
    broken = _UvLayer("UVMap", ())
    obj = _Object(
        _Mesh((broken,), loop_count=12, active=broken),
        nodes=(_UvMapNode("UVMap"),),
    )

    with pytest.raises(SourceUvIntegrityError, match="malformed required UV"):
        resolve_readable_source_uv_layer_names(obj, _settings())


def test_source_uv_fingerprint_detects_coordinate_mutation():
    layer = _UvLayer("UVMap", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    obj = _Object(_Mesh((layer,), active=layer))
    before = capture_source_uv_fingerprint(obj)

    require_source_uv_unchanged(before, obj)
    layer.uv[1].vector = (0.25, 0.75)

    with pytest.raises(SourceUvMutationError, match="changed the source Mesh UV state"):
        require_source_uv_unchanged(before, obj)


def test_optional_fingerprint_defers_non_mesh_validation_to_typed_stage():
    opaque = object()

    assert capture_source_uv_fingerprint_if_mesh(opaque) is None
    require_source_uv_unchanged_if_captured(None, opaque)


def test_optional_fingerprint_keeps_declared_mesh_strict():
    broken_mesh_object = SimpleNamespace(type="MESH", data=None)

    with pytest.raises(SourceUvIntegrityError, match="obj.data is missing"):
        capture_source_uv_fingerprint_if_mesh(broken_mesh_object)


def test_object_mode_contract_rejects_edit_mode():
    require_object_mode(SimpleNamespace(mode="OBJECT"))

    with pytest.raises(ObjectModeRequiredError, match="Finish or cancel Edit Mode"):
        require_object_mode(SimpleNamespace(mode="EDIT_MESH"))
