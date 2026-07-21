from copy import deepcopy
from math import inf, nan

import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    Bone,
    Skin,
    Slot,
    SpineDocument,
    SpineSerializer,
    SpineValidator,
)


def build_document(events):
    return SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"),),
        slots=(Slot("slot", "root"),),
        skins=(Skin("default", {}),),
        animations={"animation": {}},
        events=events,
    )


@pytest.mark.parametrize(
    "events",
    (
        {},
        {"hit": {}},
        {
            "hit": {
                "int": 1,
                "float": 2.5,
                "string": "three",
                "audio": "hit.wav",
                "volume": 0.9,
                "balance": -0.25,
            }
        },
        {"future event": {"futureField": {"enabled": True}}},
    ),
)
def test_valid_event_definitions_and_future_fields_are_accepted(events):
    document = build_document(events)

    assert SpineValidator().validate(document) == ()
    assert document.events == events


@pytest.mark.parametrize("value", (None, True, 1, "event", (), []))
def test_event_definition_must_be_a_mapping(value):
    with pytest.raises(TypeError) as error:
        build_document({"event with spaces": value})

    assert 'document.events["event with spaces"] must be a mapping' in str(
        error.value
    )


@pytest.mark.parametrize("value", (True, 1.0, "1", None))
def test_event_int_requires_a_strict_integer(value):
    with pytest.raises(TypeError, match=r"document\.events\.event\.int must be int"):
        build_document({"event": {"int": value}})


@pytest.mark.parametrize("value", (-(2**31), 2**31 - 1))
def test_event_int_accepts_signed_32_bit_boundaries(value):
    document = build_document({"event": {"int": value}})

    assert document.events["event"]["int"] == value


@pytest.mark.parametrize("value", (-(2**31) - 1, 2**31))
def test_event_int_rejects_values_outside_runtime_range(value):
    with pytest.raises(ValueError, match="signed 32-bit range"):
        build_document({"event": {"int": value}})


@pytest.mark.parametrize("field_name", ("string", "audio"))
@pytest.mark.parametrize("value", (True, 1, None, (), []))
def test_event_string_fields_require_strings(field_name, value):
    with pytest.raises(TypeError) as error:
        build_document({"event": {field_name: value}})

    assert f"document.events.event.{field_name} must be str" in str(error.value)


@pytest.mark.parametrize("field_name", ("float", "volume", "balance"))
@pytest.mark.parametrize("value", (True, "1", None, (), []))
def test_event_numeric_fields_require_numbers(field_name, value):
    with pytest.raises(TypeError) as error:
        build_document({"event": {field_name: value}})

    assert (
        f"document.events.event.{field_name} must be a finite number"
        in str(error.value)
    )


@pytest.mark.parametrize("field_name", ("float", "volume", "balance"))
@pytest.mark.parametrize("value", (nan, inf, -inf))
def test_event_numeric_fields_reject_non_finite_values(field_name, value):
    with pytest.raises(ValueError) as error:
        build_document({"event": {field_name: value}})

    assert f"document.events.event.{field_name}" in str(error.value)


def test_event_volume_and_balance_are_not_given_unproven_ranges():
    events = {
        "audio": {
            "audio": "hit.wav",
            "volume": 1.5,
            "balance": -2.0,
        }
    }

    assert build_document(events).events == events


def test_serializer_preserves_events_without_inserting_defaults():
    events = {
        "hit": {"int": 7, "futureField": True},
        "audio": {"audio": "hit.wav", "balance": -0.25},
    }
    source = deepcopy(events)

    serialized = SpineSerializer().to_dict(build_document(events))

    assert serialized["events"] == source
    assert serialized["events"]["hit"] == {
        "int": 7,
        "futureField": True,
    }
    assert "float" not in serialized["events"]["hit"]
    assert "volume" not in serialized["events"]["audio"]
