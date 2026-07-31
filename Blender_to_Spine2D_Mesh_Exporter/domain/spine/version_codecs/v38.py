"""Spine 3.8 JSON codec for legacy combined transform-constraint mixes.

Spine 3.8 uses the legacy bone ``transform`` field and separate root IK/transform
collections, but exposes one combined translation mix and one combined scale mix.
Explicit X/Y values must agree before they can be represented exactly. The codec never
averages or silently drops a channel.
"""

from __future__ import annotations

import json
from math import isfinite
from typing import Any

from ..model import SpineDocument
from ..version_target import SpineJsonTarget
from .base import SpineJsonCodecContext
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

        self._remove_unsupported_bone_ui_fields(output)
        self._rewrite_setup_transform_mixes(output)
        self._rewrite_animation_transform_mixes(output)
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


__all__ = ["Spine38JsonCodec"]
