from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.view_layer_contract import (
    ViewLayerContractError,
    source_layer_collection_states,
    validate_source_view_layer_for_camera_projection,
)


class FakeLayerCollection:
    def __init__(
        self,
        collection,
        *,
        children=(),
        exclude=False,
        holdout=False,
        indirect_only=False,
    ):
        self.collection = collection
        self.name = collection.name
        self.children = tuple(children)
        self.exclude = exclude
        self.holdout = holdout
        self.indirect_only = indirect_only


def _collection(name):
    return SimpleNamespace(name=name)


def _view_layer(source, child):
    root_collection = _collection("Scene Collection")
    root = FakeLayerCollection(root_collection, children=(child,))
    return SimpleNamespace(name="ViewLayer", objects=(source,), layer_collection=root)


def test_direct_source_collection_is_renderable():
    collection = _collection("Source Collection")
    source = SimpleNamespace(name="Source", users_collection=(collection,))
    view_layer = _view_layer(source, FakeLayerCollection(collection))

    states = validate_source_view_layer_for_camera_projection(source, view_layer)

    assert len(states) == 1
    assert states[0].direct_camera_renderable


@pytest.mark.parametrize("flag", ("exclude", "holdout", "indirect_only"))
def test_non_direct_source_collection_is_rejected(flag):
    collection = _collection("Source Collection")
    source = SimpleNamespace(name="Source", users_collection=(collection,))
    view_layer = _view_layer(
        source,
        FakeLayerCollection(collection, **{flag: True}),
    )

    with pytest.raises(ViewLayerContractError, match="excluded, Holdout, or Indirect"):
        validate_source_view_layer_for_camera_projection(source, view_layer)


def test_source_excluded_from_view_layer_is_rejected_before_collection_scan():
    collection = _collection("Source Collection")
    source = SimpleNamespace(name="Source", users_collection=(collection,))
    view_layer = _view_layer(source, FakeLayerCollection(collection))
    view_layer.objects = ()

    with pytest.raises(ViewLayerContractError, match="excluded from active View Layer"):
        validate_source_view_layer_for_camera_projection(source, view_layer)


def test_one_direct_link_allows_other_holdout_link():
    holdout_collection = _collection("Holdout")
    direct_collection = _collection("Direct")
    source = SimpleNamespace(
        name="Source",
        users_collection=(holdout_collection, direct_collection),
    )
    root = FakeLayerCollection(
        _collection("Scene Collection"),
        children=(
            FakeLayerCollection(holdout_collection, holdout=True),
            FakeLayerCollection(direct_collection),
        ),
    )
    view_layer = SimpleNamespace(name="ViewLayer", objects=(source,), layer_collection=root)

    states = source_layer_collection_states(source, view_layer)
    validated = validate_source_view_layer_for_camera_projection(source, view_layer)

    assert states == validated
    assert any(state.direct_camera_renderable for state in states)
    assert any(state.holdout for state in states)
