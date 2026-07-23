from pathlib import Path
from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.material_output_transaction import (
    MaterialOutputTransactionError,
    preserve_material_output_state,
)


ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "blender_adapter"


class _Socket:
    def __init__(self, name):
        self.name = name
        self.links = []


class _SocketCollection(list):
    def get(self, name):
        return next((socket for socket in self if socket.name == name), None)


class _Node:
    def __init__(self, name, node_type, *, active=False, surface=None):
        self.name = name
        self.type = node_type
        self.is_active_output = active
        self.inputs = _SocketCollection(() if surface is None else (surface,))


class _Link:
    def __init__(self, source, target):
        self.from_socket = source
        self.to_socket = target


class _Links:
    def __init__(self):
        self.values = []

    def new(self, source, target):
        link = _Link(source, target)
        self.values.append(link)
        target.links.append(link)
        return link

    def remove(self, link):
        self.values.remove(link)
        link.to_socket.links.remove(link)


class _NodeTree:
    def __init__(self, nodes):
        self.nodes = list(nodes)
        self.links = _Links()


class _Material:
    def __init__(self, node_tree):
        self.node_tree = node_tree


def _fixture():
    cycles_surface = _Socket("Surface")
    eevee_surface = _Socket("Surface")
    cycles = _Node(
        "Cycles Output",
        "OUTPUT_MATERIAL",
        active=True,
        surface=cycles_surface,
    )
    eevee = _Node(
        "Eevee Output",
        "OUTPUT_MATERIAL",
        active=False,
        surface=eevee_surface,
    )
    tree = _NodeTree((cycles, eevee))
    original_source = _Socket("Original Shader")
    temporary_source = _Socket("Temporary Emission")
    original_link = tree.links.new(original_source, cycles_surface)
    return SimpleNamespace(
        material=_Material(tree),
        tree=tree,
        cycles=cycles,
        eevee=eevee,
        cycles_surface=cycles_surface,
        original_source=original_source,
        temporary_source=temporary_source,
        original_link=original_link,
    )


def _surface_sources(surface):
    return tuple(link.from_socket for link in surface.links)


def test_transaction_restores_links_and_active_flags_after_partial_failure():
    fixture = _fixture()

    with pytest.raises(RuntimeError, match="partial removal"):
        with preserve_material_output_state((fixture.material,)):
            fixture.cycles.is_active_output = False
            fixture.eevee.is_active_output = True
            fixture.tree.links.remove(fixture.original_link)
            fixture.tree.links.new(
                fixture.temporary_source,
                fixture.cycles_surface,
            )
            raise RuntimeError("partial removal")

    assert _surface_sources(fixture.cycles_surface) == (fixture.original_source,)
    assert fixture.cycles.is_active_output is True
    assert fixture.eevee.is_active_output is False


def test_transaction_is_idempotent_when_inner_owner_already_restored_state():
    fixture = _fixture()

    with preserve_material_output_state((fixture.material,)):
        pass

    assert _surface_sources(fixture.cycles_surface) == (fixture.original_source,)
    assert fixture.cycles.is_active_output is True
    assert fixture.eevee.is_active_output is False


def test_transaction_rejects_material_without_output():
    material = _Material(_NodeTree(()))

    with pytest.raises(MaterialOutputTransactionError, match="no Material Output"):
        with preserve_material_output_state((material,)):
            pass


def test_scene_owner_captures_output_state_before_proxy_preparation():
    source = (ADAPTER / "scene_material_preparation.py").read_text(encoding="utf-8")

    snapshot_position = source.index("with preserve_material_output_state(materials):")
    preparation_position = source.index("with temporary_prepare_material_pass(")
    assert snapshot_position < preparation_position
    assert "_prepare_proxy_material" not in source
    assert "_restore_mutation" not in source
