"""Spine 3.8 codec extension for rigid camera-relative two-axis documents.

The historical Spine 3.8 two-axis topology splits one public Scale control into an
internal position phase before Rotation X and a geometry phase after Rotation Y. That
split is correct for model-space rigs, but it is invalid for camera-relative rigs: an
internal position phase would scale the orbital hierarchy and change the object's
camera distance.

This codec keeps the established Spine 3.8 serialization and model-space validator from
``v38``. It adds one strictly detected alternative topology for PREPROJECTED_SCREEN
outputs:

    Rotation X -> IK -> Depth -> Rotation Y -> object-base Scale

No JSON repair is performed. The typed target finalizer must already have produced the
correct bridges, weighted-bone remap, constraint order, and object-base Scale target.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .v38 import (
    _SPINE38_BRIDGE_PATTERN,
    _TWO_AXIS_DEPTH_SUFFIX,
    _constraint_bones,
    _constraint_order,
    _named_records,
    _require_dict,
    _require_list,
    _require_scale_only_payload,
    Spine38JsonCodec,
)


_POSITION_SUFFIX = "_scale_spine38_position"
_ROTATION_X_SUFFIX = "_rotation_X_constraint"
_IK_SUFFIX = "_IK"
_ROTATION_Y_SUFFIX = "_rotation_Y"
_SCALE_SUFFIX = "_scale"
_BRIDGE_SUFFIX = "_spine41_bridge"


def _detected_two_axis_prefixes(
    transform_by_name: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    """Return deterministic prefixes owning a two-axis depth constraint."""

    return tuple(
        sorted(
            {
                name[: -len(_TWO_AXIS_DEPTH_SUFFIX)]
                for name in transform_by_name
                if name.endswith(_TWO_AXIS_DEPTH_SUFFIX)
                and len(name) > len(_TWO_AXIS_DEPTH_SUFFIX)
            },
            key=lambda value: (value.casefold(), value),
        )
    )


def _require_common_inventory(
    prefix: str,
    *,
    transform_by_name: dict[str, dict[str, Any]],
    ik_by_name: dict[str, dict[str, Any]],
) -> None:
    """Require fields shared by model-space and camera-relative 3.8 rigs."""

    required_transform = (
        f"{prefix}{_ROTATION_X_SUFFIX}",
        f"{prefix}{_TWO_AXIS_DEPTH_SUFFIX}",
        f"{prefix}{_ROTATION_Y_SUFFIX}",
        f"{prefix}{_SCALE_SUFFIX}",
    )
    missing_transform = tuple(
        name for name in required_transform if name not in transform_by_name
    )
    ik_name = f"{prefix}{_IK_SUFFIX}"
    missing = missing_transform + (() if ik_name in ik_by_name else (ik_name,))
    if missing:
        raise ValueError(
            "Spine 3.8 two-axis constraint inventory is incomplete for "
            f"{prefix!r}: missing={missing}"
        )


def _bone_topology(
    output: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, tuple[str, ...]],
    dict[str, tuple[tuple[str, str], ...]],
]:
    """Index unique bones, children, and generated depth bridges."""

    bones = _require_list(output.get("bones"), path="document.bones")
    bone_by_name: dict[str, dict[str, Any]] = {}
    mutable_children: dict[str, list[str]] = {}
    mutable_bridges: dict[str, list[tuple[str, str]]] = {}

    for index, value in enumerate(bones):
        path = f"document.bones[{index}]"
        bone = _require_dict(value, path=path)
        name = bone.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}.name must be a non-empty string")
        if name in bone_by_name:
            raise ValueError(f"document.bones contains duplicate name {name!r}")
        bone_by_name[name] = bone

        parent = bone.get("parent")
        if parent is not None:
            if not isinstance(parent, str) or not parent.strip():
                raise ValueError(f"{path}.parent must be a non-empty string")
            mutable_children.setdefault(parent, []).append(name)

        match = _SPINE38_BRIDGE_PATTERN.fullmatch(name)
        if match is not None:
            prefix = match.group("prefix")
            wrapper_name = name[: -len(_BRIDGE_SUFFIX)]
            mutable_bridges.setdefault(prefix, []).append((wrapper_name, name))

    children = {
        parent: tuple(names)
        for parent, names in mutable_children.items()
    }
    bridges = {
        prefix: tuple(records)
        for prefix, records in mutable_bridges.items()
    }
    return bone_by_name, children, bridges


def _validate_camera_relative_prefix(
    prefix: str,
    *,
    transform_by_name: dict[str, dict[str, Any]],
    ik_by_name: dict[str, dict[str, Any]],
    bone_by_name: dict[str, dict[str, Any]],
    children_by_parent: dict[str, tuple[str, ...]],
    bridge_records: dict[str, tuple[tuple[str, str], ...]],
) -> None:
    """Validate one rigid camera layer without accepting malformed model-space rigs."""

    position_name = f"{prefix}{_POSITION_SUFFIX}"
    if position_name in transform_by_name:
        raise ValueError(
            f"Camera-relative Spine 3.8 rig {prefix!r} must not contain "
            f"{position_name!r}"
        )

    raw_bridges = bridge_records.get(prefix, ())
    if len(raw_bridges) != 1:
        raise ValueError(
            f"Camera-relative Spine 3.8 rig {prefix!r} must contain exactly one "
            f"depth bridge, found {len(raw_bridges)}"
        )
    wrapper_name, bridge_name = raw_bridges[0]
    wrapper = bone_by_name.get(wrapper_name)
    bridge = bone_by_name.get(bridge_name)
    if wrapper is None or bridge is None:
        raise ValueError(
            f"Camera-relative Spine 3.8 bridge topology is incomplete for {prefix!r}"
        )

    expected_orbit_parent = f"{prefix}_rotate_X"
    if bridge.get("parent") != expected_orbit_parent:
        raise ValueError(
            f"Camera-relative bridge {bridge_name!r} must be parented to "
            f"{expected_orbit_parent!r}"
        )
    if wrapper.get("parent") != bridge_name:
        raise ValueError(
            f"Camera-relative wrapper {wrapper_name!r} must be parented to "
            f"{bridge_name!r}"
        )

    layer_children = tuple(children_by_parent.get(wrapper_name, ()))
    if len(layer_children) != 1:
        raise ValueError(
            f"Camera-relative wrapper {wrapper_name!r} must have exactly one "
            f"final layer child, found {len(layer_children)}"
        )
    layer_name = layer_children[0]

    base = bone_by_name.get(prefix)
    if base is None:
        raise ValueError(
            f"Camera-relative Spine 3.8 rig {prefix!r} is missing its object base bone"
        )
    if base.get("parent") != layer_name:
        raise ValueError(
            f"Camera-relative object base {prefix!r} must be parented to final "
            f"layer {layer_name!r}; actual={base.get('parent')!r}"
        )

    rotation_x_name = f"{prefix}{_ROTATION_X_SUFFIX}"
    ik_name = f"{prefix}{_IK_SUFFIX}"
    depth_name = f"{prefix}{_TWO_AXIS_DEPTH_SUFFIX}"
    rotation_y_name = f"{prefix}{_ROTATION_Y_SUFFIX}"
    scale_name = f"{prefix}{_SCALE_SUFFIX}"

    rotation_x = transform_by_name[rotation_x_name]
    scale_ik = ik_by_name[ik_name]
    depth = transform_by_name[depth_name]
    rotation_y = transform_by_name[rotation_y_name]
    scale = transform_by_name[scale_name]

    base_order = _constraint_order(
        rotation_x,
        path=f"document.transform[{rotation_x_name}]",
    )
    actual_orders = (
        base_order,
        _constraint_order(scale_ik, path=f"document.ik[{ik_name}]"),
        _constraint_order(depth, path=f"document.transform[{depth_name}]"),
        _constraint_order(
            rotation_y,
            path=f"document.transform[{rotation_y_name}]",
        ),
        _constraint_order(scale, path=f"document.transform[{scale_name}]"),
    )
    expected_orders = tuple(range(base_order, base_order + 5))
    if actual_orders != expected_orders:
        raise ValueError(
            f"Camera-relative Spine 3.8 constraints for {prefix!r} must evaluate "
            "as X/IK/Depth/Y/ObjectScale in one dense block; "
            f"expected={expected_orders}, actual={actual_orders}"
        )

    depth_bones = _constraint_bones(
        depth,
        path=f"document.transform[{depth_name}]",
    )
    if depth_bones != (wrapper_name,):
        raise ValueError(
            f"Camera-relative depth constraint {depth_name!r} must constrain only "
            f"wrapper {wrapper_name!r}; actual={depth_bones}"
        )

    rotation_y_bones = _constraint_bones(
        rotation_y,
        path=f"document.transform[{rotation_y_name}]",
    )
    if rotation_y_bones != (layer_name,):
        raise ValueError(
            f"Camera-relative Rotation Y constraint {rotation_y_name!r} must "
            f"constrain only layer {layer_name!r}; actual={rotation_y_bones}"
        )

    scale_bones = _constraint_bones(
        scale,
        path=f"document.transform[{scale_name}]",
    )
    if scale_bones != (prefix,):
        raise ValueError(
            f"Camera-relative Scale constraint {scale_name!r} must constrain only "
            f"object base {prefix!r}; actual={scale_bones}"
        )
    expected_target = f"{prefix}_scale"
    if scale.get("target") != expected_target:
        raise ValueError(
            f"Camera-relative Scale constraint {scale_name!r} must target "
            f"{expected_target!r}; actual={scale.get('target')!r}"
        )
    _require_scale_only_payload(
        scale,
        path=f"document.transform[{scale_name}]",
    )


def _model_space_validation_view(
    output: dict[str, Any],
    *,
    camera_prefixes: frozenset[str],
) -> dict[str, Any]:
    """Remove validated camera rigs from a copy used by the model-space validator."""

    filtered = deepcopy(output)
    camera_transform_names = {
        name
        for prefix in camera_prefixes
        for name in (
            f"{prefix}{_ROTATION_X_SUFFIX}",
            f"{prefix}{_TWO_AXIS_DEPTH_SUFFIX}",
            f"{prefix}{_ROTATION_Y_SUFFIX}",
            f"{prefix}{_SCALE_SUFFIX}",
        )
    }
    camera_ik_names = {f"{prefix}{_IK_SUFFIX}" for prefix in camera_prefixes}

    transforms = _require_list(
        filtered.get("transform", []),
        path="document.transform",
    )
    filtered["transform"] = [
        value
        for value in transforms
        if not (
            isinstance(value, dict)
            and value.get("name") in camera_transform_names
        )
    ]

    ik_values = _require_list(filtered.get("ik", []), path="document.ik")
    filtered["ik"] = [
        value
        for value in ik_values
        if not (
            isinstance(value, dict)
            and value.get("name") in camera_ik_names
        )
    ]

    bones = _require_list(filtered.get("bones"), path="document.bones")
    filtered["bones"] = [
        value
        for value in bones
        if not (
            isinstance(value, dict)
            and isinstance(value.get("name"), str)
            and (
                (match := _SPINE38_BRIDGE_PATTERN.fullmatch(value["name"]))
                is not None
            )
            and match.group("prefix") in camera_prefixes
        )
    ]
    return filtered


class Spine38CameraRelativeJsonCodec(Spine38JsonCodec):
    """Serialize Spine 3.8 while validating both approved two-axis topologies."""

    @staticmethod
    def _validate_two_axis_runtime_topology(output: dict[str, Any]) -> None:
        if not isinstance(output, dict):
            raise TypeError("output must be a JSON object")

        transform_by_name = _named_records(output, "transform")
        ik_by_name = _named_records(output, "ik")
        prefixes = _detected_two_axis_prefixes(transform_by_name)
        if not prefixes:
            Spine38JsonCodec._validate_two_axis_runtime_topology(output)
            return

        for prefix in prefixes:
            _require_common_inventory(
                prefix,
                transform_by_name=transform_by_name,
                ik_by_name=ik_by_name,
            )

        camera_prefixes = frozenset(
            prefix
            for prefix in prefixes
            if f"{prefix}{_POSITION_SUFFIX}" not in transform_by_name
        )
        if not camera_prefixes:
            Spine38JsonCodec._validate_two_axis_runtime_topology(output)
            return

        bone_by_name, children_by_parent, bridge_records = _bone_topology(output)
        expected_bridge_prefixes = set(prefixes)
        actual_bridge_prefixes = set(bridge_records)
        if actual_bridge_prefixes != expected_bridge_prefixes:
            raise ValueError(
                "Spine 3.8 two-axis documents must pass target finalization before "
                "serialization; bridge topology differs from constraint inventory: "
                f"missing_bridges={tuple(sorted(expected_bridge_prefixes - actual_bridge_prefixes))}, "
                f"unexpected_bridges={tuple(sorted(actual_bridge_prefixes - expected_bridge_prefixes))}"
            )

        for prefix in sorted(camera_prefixes, key=lambda value: (value.casefold(), value)):
            _validate_camera_relative_prefix(
                prefix,
                transform_by_name=transform_by_name,
                ik_by_name=ik_by_name,
                bone_by_name=bone_by_name,
                children_by_parent=children_by_parent,
                bridge_records=bridge_records,
            )

        model_prefixes = set(prefixes) - set(camera_prefixes)
        if model_prefixes:
            Spine38JsonCodec._validate_two_axis_runtime_topology(
                _model_space_validation_view(
                    output,
                    camera_prefixes=camera_prefixes,
                )
            )


__all__ = ["Spine38CameraRelativeJsonCodec"]
