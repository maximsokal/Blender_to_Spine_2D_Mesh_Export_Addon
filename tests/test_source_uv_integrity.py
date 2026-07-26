from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.source_uv_integrity import (
    ObjectModeRequiredError,
    SourceUvIntegrityError,
    SourceUvMutationError,
    capture_source_uv_fingerprint,
    require_object_mode,
    require_source_uv_unchanged,
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


class _Object:
    type = "MESH"

    def __init__(self, mesh, material_slots=()):
        self.data = mesh
        self.material_slots = tuple(material_slots)


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


def test_source_uv_fingerprint_detects_coordinate_mutation():
    layer = _UvLayer("UVMap", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    obj = _Object(_Mesh((layer,), active=layer))
    before = capture_source_uv_fingerprint(obj)

    require_source_uv_unchanged(before, obj)
    layer.uv[1].vector = (0.25, 0.75)

    with pytest.raises(SourceUvMutationError, match="changed the source Mesh UV state"):
        require_source_uv_unchanged(before, obj)


def test_object_mode_contract_rejects_edit_mode():
    require_object_mode(SimpleNamespace(mode="OBJECT"))

    with pytest.raises(ObjectModeRequiredError, match="Finish or cancel Edit Mode"):
        require_object_mode(SimpleNamespace(mode="EDIT_MESH"))
