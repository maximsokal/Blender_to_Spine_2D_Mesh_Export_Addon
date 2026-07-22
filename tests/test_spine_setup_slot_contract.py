from dataclasses import FrozenInstanceError

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.setup_slot_contract import (
    SetupSlotIndex,
    resolve_setup_slot_index,
)


def test_index_retains_exact_tuple_and_returns_setup_positions():
    slot_names = ("front", "back")

    index = SetupSlotIndex(slot_names)

    assert index.slot_names is slot_names
    assert index.require("front", path="document.animations.idle.slots.front") == 0
    assert index.require("back", path="document.animations.idle.slots.back") == 1


def test_index_is_frozen_and_does_not_expose_mutable_lookup_state():
    index = SetupSlotIndex(("slot",))

    with pytest.raises(FrozenInstanceError):
        index.slot_names = ("changed",)

    assert not isinstance(index._index_by_name, dict)
    assert isinstance(index._ambiguous_names, frozenset)


@pytest.mark.parametrize("slot_names", ([], "slot", {"slot"}, None))
def test_slot_names_must_be_an_exact_tuple(slot_names):
    with pytest.raises(TypeError, match="slot_names must be tuple"):
        SetupSlotIndex(slot_names)


@pytest.mark.parametrize("slot_name", (None, True, 1, (), {}))
def test_setup_slot_names_require_strings(slot_name):
    with pytest.raises(TypeError, match=r"slot_names\[0\] must be str"):
        SetupSlotIndex((slot_name,))


@pytest.mark.parametrize("slot_name", ("", " ", "\t"))
def test_setup_slot_names_cannot_be_empty(slot_name):
    with pytest.raises(ValueError, match=r"slot_names\[0\] cannot be empty"):
        SetupSlotIndex((slot_name,))


def test_lookup_rejects_undefined_and_duplicated_slots_with_caller_path():
    path = "document.animations.idle.slots.item"

    with pytest.raises(ValueError, match="undefined slot 'item'") as undefined:
        SetupSlotIndex(("slot",)).require("item", path=path)
    assert path in str(undefined.value)

    with pytest.raises(
        ValueError,
        match="duplicated setup slot 'item'",
    ) as duplicated:
        SetupSlotIndex(("item", "item")).require("item", path=path)
    assert path in str(duplicated.value)


def test_lookup_validates_reference_name_and_path():
    index = SetupSlotIndex(("slot",))

    with pytest.raises(TypeError, match="slot name must be str"):
        index.require(None, path="document.animations.idle.slots.null")
    with pytest.raises(ValueError, match="slot name cannot be empty"):
        index.require(" ", path="document.animations.idle.slots.empty")
    with pytest.raises(ValueError, match="path must be a non-empty string"):
        index.require("slot", path="")


def test_resolver_builds_direct_index_and_reuses_exact_index():
    slot_names = ("slot",)

    direct = resolve_setup_slot_index(slot_names, None)
    reused = resolve_setup_slot_index(slot_names, direct)

    assert reused is direct
    assert direct.slot_names is slot_names


def test_resolver_rejects_wrong_type_and_equivalent_stale_tuple():
    slot_names = ("slot",)

    with pytest.raises(
        TypeError,
        match="setup_slot_index must be SetupSlotIndex or None",
    ):
        resolve_setup_slot_index(slot_names, object())

    equivalent_but_distinct = tuple(["slot"])
    stale = SetupSlotIndex(equivalent_but_distinct)
    assert equivalent_but_distinct == slot_names
    assert equivalent_but_distinct is not slot_names

    with pytest.raises(ValueError, match="exact slot_names tuple"):
        resolve_setup_slot_index(slot_names, stale)
