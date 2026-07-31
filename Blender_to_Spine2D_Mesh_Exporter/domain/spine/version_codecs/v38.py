"""Spine 3.8 JSON codec for legacy mixes and runtime-safe constraint order.

Spine 3.8 uses the legacy bone ``transform`` field and separate root IK/transform
collections, but exposes one combined translation mix and one combined scale mix.
Explicit X/Y values must agree before they can be represented exactly. The codec never
averages or silently drops a channel.
"""

from __future__ import annotations

import json
from math import isfinite
import re
from typing import Any

from ..model import SpineDocument
from ..version_target import SpineJsonTarget
from .base import SpineJsonCodecContext
from .runtime_constraint_order import normalize_runtime_constraint_orders
from .v40 import _sequence_paths
from .v41 import Spine41JsonCodec


_NEW_MIX_FIELDS = frozenset(
    {
        "mixRotate",
        "mixX",
        "mixY",
        "mixScaleX",
        "mixScaleY",
        "mixShearY",
    }
)
_LEGACY_MIX_FIELDS = frozenset(
    {"rotateMix", "translateMix", "scaleMix", "shearMix"}
)
_SPINE38_BRIDGE_PATTERN = re.compile(
    r"^(?P<prefix>.+)_(?P<layer>[0-9]+)_scale_spine41_bridge$"
)
_TWO_AXIS_DEPTH_SUFFIX = "_scale_rotate_X_constraint"


def _require_dict(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a JSON object")
    return value


def _require_list(value: Any, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a JSON array")
    return value


def _finite_mix(value: Any, *, path: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{path} must be a finite number")
    if not isfinite(float(value)):
        raise ValueError(f"{path} must be finite")
    return value


def _mix_value(
    mapping: dict[str, Any],
    field_name: str,
    *,
    default: float,
    path: str,
) -> float | int:
    return _finite_mix(
        mapping.get(field_name, default),
        path=f"{path}.{field_name}",
    )


def _combined_mix(
    mapping: dict[str, Any],
    x_field: str,
    y_field: str,
    *,
    default: float,
    path: str,
    legacy_name: str,
) -> float | int:
    """Resolve one 4.2 X/Y pair using the official inherited-Y default."""

    x_value = _mix_value(mapping, x_field, default=default, path=path)
    y_value = _finite_mix(
        mapping.get(y_field, x_value),
        path=f"{path}.{y_field}",
    )
    if float(x_value) != float(y_value):
        raise ValueError(
            f"{path} cannot be represented by Spine 3.8 {legacy_name}: "
            f"{x_field}={x_value!r}, {y_field}={y_value!r}"
        )
    return x_value


def _rewrite_mix_mapping(mapping: dict[str, Any], *, path: str) -> None:
    collision = tuple(sorted(_LEGACY_MIX_FIELDS.intersection(mapping)))
    if collision:
        raise ValueError(
            f"{path} already contains Spine 3.8 mix fields reserved for the codec: "
            f"{collision}"
        )

    rotate_mix = _mix_value(mapping, "mixRotate", default=1.0, path=path)
    translate_mix = _combined_mix(
        mapping,
        "mixX",
        "mixY",
        default=1.0,
        path=path,
        legacy_name="translateMix",
    )
    scale_mix = _combined_mix(
        mapping,
        "mixScaleX",
        "mixScaleY",
        default=1.0,
        path=path,
        legacy_name="scaleMix",
    )
    shear_mix = _mix_value(mapping, "mixShearY", default=1.0, path=path)

    for field_name in _NEW_MIX_FIELDS:
        mapping.pop(field_name, None)
    mapping["rotateMix"] = rotate_mix
    mapping["translateMix"] = translate_mix
    mapping["scaleMix"] = scale_mix
    mapping["shearMix"] = shear_mix


def _named_records(
    output: dict[str, Any],
    collection_name: str,
) -> dict[str, dict[str, Any]]:
    values = _require_list(
        output.get(collection_name, []),
        path=f"document.{collection_name}",
    )
    records: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        path = f"document.{collection_name}[{index}]"
        record = _require_dict(value, path=path)
        name = record.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{path}.name must be a non-empty string")
        if name in records:
            raise ValueError(
                f"document.{collection_name} contains duplicate name {name!r}"
            )
        records[name] = record
    return records


def _constraint_order(record: dict[str, Any], *, path: str) -> int:
    value = record.get("order", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path}.order must be a non-negative integer")
    return value


def _constraint_bones(record: dict[str, Any], *, path: str) -> tuple[str, ...]:
    values = _require_list(record.get("bones"), path=f"{path}.bones")
    result: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path}.bones[{index}] must be a non-empty string")
        result.append(value)
    if not result:
        raise ValueError(f"{path}.bones cannot be empty")
    if len(result) != len(set(result)):
        raise ValueError(f"{path}.bones cannot contain duplicates")
    return tuple(result)


def _require_scale_only_payload(record: dict[str, Any], *, path: str) -> None:
    expected = {
        "rotateMix": 0.0,
        "translateMix": 0.0,
        "scaleMix": 1.0,
        "shearMix": 0.0,
    }
    for field_name, expected_value in expected.items():
        actual = _finite_mix(record.get(field_name), path=f"{path}.{field_name}")
        if float(actual) != expected_value:
            raise ValueError(
                f"{path}.{field_name} must be {expected_value}, got {actual!r}"
            )
    if record.get("relative") is not True:
        raise ValueError(f"{path}.relative must be true")
    if record.get("local") not in {None, False}:
        raise ValueError(f"{path}.local must be false or omitted")


def _two_axis_prefixes(
    transform_by_name: dict[str, dict[str, Any]],
    ik_by_name: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    """Detect complete 2-Axis rigs by their target-specific depth constraint name."""

    prefixes = tuple(
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
    for prefix in prefixes:
        required_transform = (
            f"{prefix}_rotation_X_constraint",
            f"{prefix}_scale_spine38_position",
            f"{prefix}_scale_rotate_X_constraint",
            f"{prefix}_rotation_Y",
            f"{prefix}_scale",
        )
        missing_transform = tuple(
            name for name in required_transform if name not in transform_by_name
        )
        ik_name = f"{prefix}_IK"
        if missing_transform or ik_name not in ik_by_name:
            missing = missing_transform + (() if ik_name in ik_by_name else (ik_name,))
            raise ValueError(
                f"Spine 3.8 two-axis constraint inventory is incomplete for "
                f"{prefix!r}: missing={missing}"
            )
    return prefixes


class Spine38JsonCodec(Spine41JsonCodec):
    """Translate canonical Spine 4.2-shaped documents to exact Spine 3.8.99 JSON."""

    @property
    def target(self) -> SpineJsonTarget:
        return SpineJsonTarget.SPINE_3_8

    def to_json(
        self,
        document: SpineDocument,
        *,
        context: SpineJsonCodecContext,
        indent: int = 2,
    ) -> str:
        if not isinstance(document, SpineDocument):
            raise TypeError("document must be SpineDocument")
        if not isinstance(context, SpineJsonCodecContext):
            raise TypeError("context must be SpineJsonCodecContext")
        if context.target is not self.target:
            raise ValueError(
                f"Spine38JsonCodec requires {self.target.value}, "
                f"got {context.target.value}"
            )

        encoded = super().to_json(document, context=context, indent=indent)
        output = _require_dict(json.loads(encoded), path="document")
        sequence_paths = _sequence_paths(output)
        if sequence_paths:
            raise ValueError(
                "Spine 3.8.99 does not support attachment or animation sequences; "
                f"remove sequence data before export: {sequence_paths}"
            )

        # Spine 3.8 updateCache scans phases 0..constraint_count-1. Historical
        # three-axis rigs author phases 1..6, so normalize only the detached JSON view
        # while preserving the canonical rig document and its cross-version metadata.
        normalize_runtime_constraint_orders(
            output,
            collections=("ik", "transform", "path"),
        )
        self._remove_unsupported_bone_ui_fields(output)
        self._rewrite_setup_transform_mixes(output)
        self._rewrite_animation_transform_mixes(output)
        self._validate_two_axis_runtime_topology(output)
        return json.dumps(
            output,
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    @staticmethod
    def _remove_unsupported_bone_ui_fields(output: dict[str, Any]) -> None:
        bones = _require_list(output.get("bones"), path="document.bones")
        for index, value in enumerate(bones):
            bone = _require_dict(value, path=f"document.bones[{index}]")
            bone.pop("color", None)
            bone.pop("icon", None)

    @staticmethod
    def _rewrite_setup_transform_mixes(output: dict[str, Any]) -> None:
        constraints_value = output.get("transform")
        if constraints_value is None:
            return
        constraints = _require_list(constraints_value, path="document.transform")
        for index, value in enumerate(constraints):
            constraint = _require_dict(
                value,
                path=f"document.transform[{index}]",
            )
            _rewrite_mix_mapping(
                constraint,
                path=f"document.transform[{index}]",
            )

    @staticmethod
    def _rewrite_animation_transform_mixes(output: dict[str, Any]) -> None:
        animations_value = output.get("animations")
        if animations_value is None:
            return
        animations = _require_dict(
            animations_value,
            path="document.animations",
        )
        for animation_name, animation_value in animations.items():
            animation_path = f"document.animations.{animation_name}"
            animation = _require_dict(animation_value, path=animation_path)
            timelines_value = animation.get("transform")
            if timelines_value is None:
                continue
            timelines = _require_dict(
                timelines_value,
                path=f"{animation_path}.transform",
            )
            for constraint_name, frames_value in timelines.items():
                frames_path = f"{animation_path}.transform.{constraint_name}"
                frames = _require_list(frames_value, path=frames_path)
                for frame_index, frame_value in enumerate(frames):
                    frame = _require_dict(
                        frame_value,
                        path=f"{frames_path}[{frame_index}]",
                    )
                    _rewrite_mix_mapping(
                        frame,
                        path=f"{frames_path}[{frame_index}]",
                    )

    @staticmethod
    def _validate_two_axis_runtime_topology(output: dict[str, Any]) -> None:
        """Reject any 2-Axis graph that can feed stale child matrices to Rotation Y."""

        bones = _require_list(output.get("bones"), path="document.bones")
        bone_by_name: dict[str, dict[str, Any]] = {}
        bridge_records: dict[str, list[tuple[str, str]]] = {}
        children_by_parent: dict[str, list[str]] = {}

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
                children_by_parent.setdefault(parent, []).append(name)

            match = _SPINE38_BRIDGE_PATTERN.fullmatch(name)
            if match is not None:
                prefix = match.group("prefix")
                wrapper_name = name[: -len("_spine41_bridge")]
                bridge_records.setdefault(prefix, []).append(
                    (wrapper_name, name)
                )

        ik_by_name = _named_records(output, "ik")
        transform_by_name = _named_records(output, "transform")
        prefixes = _two_axis_prefixes(transform_by_name, ik_by_name)
        if not prefixes:
            if bridge_records:
                raise ValueError(
                    "Spine 3.8 bridge bones exist without a complete two-axis "
                    f"constraint inventory: prefixes={tuple(sorted(bridge_records))}"
                )
            # Three-axis Spine 3.8 documents do not use this target topology.
            return

        expected_prefixes = set(prefixes)
        actual_bridge_prefixes = set(bridge_records)
        if actual_bridge_prefixes != expected_prefixes:
            raise ValueError(
                "Spine 3.8 two-axis documents must pass target finalization before "
                "serialization; bridge topology differs from constraint inventory: "
                f"missing_bridges={tuple(sorted(expected_prefixes - actual_bridge_prefixes))}, "
                f"unexpected_bridges={tuple(sorted(actual_bridge_prefixes - expected_prefixes))}"
            )

        for prefix in prefixes:
            raw_records = bridge_records[prefix]
            wrappers: list[str] = []
            layers: list[str] = []
            for wrapper_name, bridge_name in raw_records:
                wrapper = bone_by_name.get(wrapper_name)
                bridge = bone_by_name.get(bridge_name)
                if wrapper is None or bridge is None:
                    raise ValueError(
                        f"Spine 3.8 bridge topology is incomplete for {prefix!r}"
                    )
                if wrapper.get("parent") != bridge_name:
                    raise ValueError(
                        f"Spine 3.8 wrapper {wrapper_name!r} must be parented to "
                        f"{bridge_name!r}"
                    )
                children = tuple(children_by_parent.get(wrapper_name, ()))
                if len(children) != 1:
                    raise ValueError(
                        f"Spine 3.8 wrapper {wrapper_name!r} must have exactly one "
                        f"final layer child, found {len(children)}"
                    )
                wrappers.append(wrapper_name)
                layers.append(children[0])

            if len(wrappers) != len(set(wrappers)) or len(layers) != len(set(layers)):
                raise ValueError(
                    f"Spine 3.8 two-axis wrapper/layer mapping is ambiguous for "
                    f"{prefix!r}"
                )

            rotation_x_name = f"{prefix}_rotation_X_constraint"
            ik_name = f"{prefix}_IK"
            public_scale_name = f"{prefix}_scale"
            position_scale_name = f"{prefix}_scale_spine38_position"
            depth_name = f"{prefix}_scale_rotate_X_constraint"
            rotation_y_name = f"{prefix}_rotation_Y"
            rotation_x = transform_by_name[rotation_x_name]
            scale_ik = ik_by_name[ik_name]
            position_scale = transform_by_name[position_scale_name]
            depth_scale = transform_by_name[depth_name]
            rotation_y = transform_by_name[rotation_y_name]
            public_scale = transform_by_name[public_scale_name]

            base_order = _constraint_order(
                rotation_x,
                path=f"document.transform[{rotation_x_name}]",
            )
            actual_orders = (
                base_order,
                _constraint_order(scale_ik, path=f"document.ik[{ik_name}]"),
                _constraint_order(
                    position_scale,
                    path=f"document.transform[{position_scale_name}]",
                ),
                _constraint_order(
                    depth_scale,
                    path=f"document.transform[{depth_name}]",
                ),
                _constraint_order(
                    rotation_y,
                    path=f"document.transform[{rotation_y_name}]",
                ),
                _constraint_order(
                    public_scale,
                    path=f"document.transform[{public_scale_name}]",
                ),
            )
            expected_orders = tuple(range(base_order, base_order + 6))
            if actual_orders != expected_orders:
                raise ValueError(
                    f"Spine 3.8 two-axis constraints for {prefix!r} must evaluate "
                    "as X/IK/ScalePosition/Depth/Y/ScaleGeometry in one dense block; "
                    f"expected={expected_orders}, actual={actual_orders}"
                )

            wrapper_set = set(wrappers)
            layer_set = set(layers)
            depth_bones = _constraint_bones(
                depth_scale,
                path=f"document.transform[{depth_name}]",
            )
            if len(depth_bones) != len(wrappers) or set(depth_bones) != wrapper_set:
                raise ValueError(
                    f"Spine 3.8 depth constraint {depth_name!r} must constrain every "
                    f"wrapper exactly once; expected={tuple(wrappers)}, "
                    f"actual={depth_bones}"
                )

            position_scale_bones = _constraint_bones(
                position_scale,
                path=f"document.transform[{position_scale_name}]",
            )
            expected_collapse = f"{prefix}_scale_rotate_X"
            if position_scale_bones != (expected_collapse,):
                raise ValueError(
                    f"Spine 3.8 position Scale constraint {position_scale_name!r} "
                    f"must constrain only {expected_collapse!r}; "
                    f"actual={position_scale_bones}"
                )

            rotation_y_bones = _constraint_bones(
                rotation_y,
                path=f"document.transform[{rotation_y_name}]",
            )
            if (
                len(rotation_y_bones) != len(layers)
                or set(rotation_y_bones) != layer_set
            ):
                raise ValueError(
                    f"Spine 3.8 Rotation Y constraint {rotation_y_name!r} must own "
                    f"every final layer exactly once; expected={tuple(layers)}, "
                    f"actual={rotation_y_bones}"
                )

            public_scale_bones = _constraint_bones(
                public_scale,
                path=f"document.transform[{public_scale_name}]",
            )
            if (
                len(public_scale_bones) != len(layers)
                or set(public_scale_bones) != layer_set
            ):
                raise ValueError(
                    f"Spine 3.8 public Scale constraint {public_scale_name!r} must "
                    f"own every final layer exactly once; expected={tuple(layers)}, "
                    f"actual={public_scale_bones}"
                )
            if position_scale.get("target") != public_scale.get("target"):
                raise ValueError(
                    f"Spine 3.8 split Scale constraints for {prefix!r} must use the "
                    "same control target"
                )
            _require_scale_only_payload(
                position_scale,
                path=f"document.transform[{position_scale_name}]",
            )
            _require_scale_only_payload(
                public_scale,
                path=f"document.transform[{public_scale_name}]",
            )


__all__ = ["Spine38JsonCodec"]
