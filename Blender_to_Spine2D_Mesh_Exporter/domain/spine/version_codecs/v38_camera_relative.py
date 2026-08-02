"""Spine 3.8 codec validation for rigid camera-relative two-axis documents.

Model-space Spine 3.8 rigs use the historical six-phase constraint schedule with an
internal position Scale before Rotation X. Camera-relative PREPROJECTED_SCREEN rigs must
not use that position phase because it would scale the orbital hierarchy and change the
object's distance from camera zero.

This module keeps the established Spine 3.8 serializer and model-space validator. It
recognizes the camera-relative alternative only from its complete bone hierarchy:

    depth bridge -> depth wrapper -> final layer -> object base

The absence of ``*_scale_spine38_position`` alone is never enough to classify a rig as
camera-relative. Unfinalized or stale model-space graphs therefore continue to fail in
the historical validator with the historical diagnostics.
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
    """Require constraints shared by both approved Spine 3.8 two-axis schemas."""

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
    children: dict[str, list[str]] = {}
    bridges: dict[str, list[tuple[str, str]]] = {}

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
            children.setdefault(parent, []).append(name)

        match = _SPINE38_BRIDGE_PATTERN.fullmatch(name)
        if match is not None:
            prefix = match.group("prefix")
            wrapper_name = name[: -len(_BRIDGE_SUFFIX)]
            bridges.setdefault(prefix, []).append((wrapper_name, name))

    return (
        bone_by_name,
        {parent: tuple(names) for parent, names in children.items()},
        {prefix: tuple(records) for prefix, records in bridges.items()},
    )


def _camera_chain(
    prefix: str,
    *,
    bone_by_name: dict[str, dict[str, Any]],
    children_by_parent: dict[str, tuple[str, ...]],
    bridge_records: dict[str, tuple[tuple[str, str], ...]],
) -> tuple[str, str, str] | None:
    """Return wrapper, bridge, and layer only for the full camera hierarchy marker."""

    raw_bridges = bridge_records.get(prefix, ())
    if len(raw_bridges) != 1:
        return None

    wrapper_name, bridge_name = raw_bridges[0]
    wrapper = bone_by_name.get(wrapper_name)
    bridge = bone_by_name.get(bridge_name)
    base = bone_by_name.get(prefix)
    if wrapper is None or bridge is None or base is None:
        return None

    expected_orbit_parent = f"{prefix}_rotate_X"
    if bridge.get("parent") != expected_orbit_parent:
        return None
    if wrapper.get("parent") != bridge_name:
        return None

    layer_children = tuple(children_by_parent.get(wrapper_name, ()))
    if len(layer_children) != 1:
        return None
    layer_name = layer_children[0]
    if base.get("parent") != layer_name:
        return None

    return wrapper_name, bridge_name, layer_name


def _validate_camera_relative_prefix(
    prefix: str,
    *,
    transform_by_name: dict[str, dict[str, Any]],
    ik_by_name: dict[str, dict[str, Any]],
    bone_by_name: dict[str, dict[str, Any]],
    children_by_parent: dict[str, tuple[str, ...]],
    bridge_records: dict[str, tuple[tuple[str, str], ...]],
) -> None:
    """Validate the approved X/IK/Depth/Y/ObjectScale camera-relative schedule."""

    position_name = f"{prefix}{_POSITION_SUFFIX}"
    if position_name in transform_by_name:
        raise ValueError(
            f"Camera-relative Spine 3.8 rig {prefix!r} must not contain "
            f"{position_name!r}"
        )

    chain = _camera_chain(
        prefix,
        bone_by_name=bone_by_name,
        children_by_parent=children_by_parent,
        bridge_records=bridge_records,
    )
    if chain is None:
        raise ValueError(
            f"Camera-relative Spine 3.8 hierarchy is incomplete for {prefix!r}"
        )
    wrapper_name, _bridge_name, layer_name = chain

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
    """Remove validated camera constraints and bridges before model-space validation."""

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

        bone_by_name, children_by_parent, bridge_records = _bone_topology(output)
        camera_prefixes = frozenset(
            prefix
            for prefix in prefixes
            if f"{prefix}{_POSITION_SUFFIX}" not in transform_by_name
            and _camera_chain(
                prefix,
                bone_by_name=bone_by_name,
                children_by_parent=children_by_parent,
                bridge_records=bridge_records,
            )
            is not None
        )

        if not camera_prefixes:
            Spine38JsonCodec._validate_two_axis_runtime_topology(output)
            return

        for prefix in sorted(
            camera_prefixes,
            key=lambda value: (value.casefold(), value),
        ):
            _validate_camera_relative_prefix(
                prefix,
                transform_by_name=transform_by_name,
                ik_by_name=ik_by_name,
                bone_by_name=bone_by_name,
                children_by_parent=children_by_parent,
                bridge_records=bridge_records,
            )

        # Always run the historical validator on the remaining graph. Besides checking
        # model-space rigs, this catches unrelated or orphaned bridge bones in mixed
        # documents instead of silently ignoring them.
        Spine38JsonCodec._validate_two_axis_runtime_topology(
            _model_space_validation_view(
                output,
                camera_prefixes=camera_prefixes,
            )
        )


__all__ = ["Spine38CameraRelativeJsonCodec"]
