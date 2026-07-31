"""Spine 4.3 JSON codec for unified setup-pose constraints.

Spine 4.3 replaces the separate root ``ik`` and ``transform`` arrays with one ordered
``constraints`` array. Transform constraints also rename the source bone, split local
space selection into source/target flags, rename relative evaluation to additive, and
require an explicit source-property to target-property mapping.

This codec owns only schema translation. Rig topology and scope acceptance remain in the
builder/finalizer and capability layers.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from ..model import SpineDocument
from ..serializer import SpineSerializer
from ..version_target import SpineJsonTarget
from .base import SpineJsonCodecContext, SpineJsonVersionCodec


_TRANSFORM_PROPERTIES = (
    "rotate",
    "x",
    "y",
    "scaleX",
    "scaleY",
    "shearY",
)


def _require_dict(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a JSON object")
    return value


def _require_list(value: Any, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a JSON array")
    return value


def _require_name(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty string")
    return value


def _require_order(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{path} must be a non-negative integer")
    return value


def _extend_unique_names(
    mapping: dict[str, Any],
    key: str,
    names: Iterable[str],
    *,
    path: str,
) -> None:
    resolved = tuple(names)
    if not resolved:
        return
    existing_value = mapping.get(key)
    if existing_value is None:
        mapping[key] = list(resolved)
        return
    existing = _require_list(existing_value, path=f"{path}.{key}")
    if not all(isinstance(name, str) and name for name in existing):
        raise TypeError(f"{path}.{key} must contain non-empty strings")
    for name in resolved:
        if name not in existing:
            existing.append(name)


def _same_property_mapping() -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Return detached same-property mappings for all legacy transform channels."""

    return {
        property_name: {"to": {property_name: {}}}
        for property_name in _TRANSFORM_PROPERTIES
    }


class Spine43JsonCodec(SpineJsonVersionCodec):
    """Translate one canonical Spine 4.2-shaped document to exact Spine 4.3.23 JSON."""

    @property
    def target(self) -> SpineJsonTarget:
        return SpineJsonTarget.SPINE_4_3

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
                f"Spine43JsonCodec requires {self.target.value}, "
                f"got {context.target.value}"
            )

        canonical_json = SpineSerializer(validator=context.validator).to_json(
            document,
            indent=indent,
        )
        output = _require_dict(json.loads(canonical_json), path="document")
        self._rewrite_skeleton(output)
        self._rewrite_skin_constraint_membership(output, document)
        self._rewrite_unified_constraints(output)
        self._reject_untranslated_constraint_families(output)

        return json.dumps(
            output,
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    def _rewrite_skeleton(self, output: dict[str, Any]) -> None:
        skeleton = _require_dict(output.get("skeleton"), path="document.skeleton")
        skeleton["spine"] = self.target.exact_version

    @staticmethod
    def _rewrite_skin_constraint_membership(
        output: dict[str, Any],
        document: SpineDocument,
    ) -> None:
        ik_names = {constraint.name for constraint in document.ik}
        transform_names = {constraint.name for constraint in document.transform}
        ambiguous = ik_names.intersection(transform_names)
        if ambiguous:
            raise ValueError(
                "IK and transform constraint names must be globally unique for "
                f"Spine 4.3 skin membership: {tuple(sorted(ambiguous))}"
            )

        skins = _require_list(output.get("skins"), path="document.skins")
        for index, value in enumerate(skins):
            skin_path = f"document.skins[{index}]"
            skin = _require_dict(value, path=skin_path)
            raw_constraints = skin.pop("constraints", None)
            if raw_constraints is None:
                continue
            constraints = _require_list(
                raw_constraints,
                path=f"{skin_path}.constraints",
            )
            ik_membership: list[str] = []
            transform_membership: list[str] = []
            for item_index, raw_name in enumerate(constraints):
                name = _require_name(
                    raw_name,
                    path=f"{skin_path}.constraints[{item_index}]",
                )
                if name in ik_names:
                    ik_membership.append(name)
                elif name in transform_names:
                    transform_membership.append(name)
                else:
                    raise ValueError(
                        f"{skin_path}.constraints references unsupported or unknown "
                        f"constraint {name!r} for Spine 4.3"
                    )
            _extend_unique_names(
                skin,
                "ik",
                ik_membership,
                path=skin_path,
            )
            _extend_unique_names(
                skin,
                "transform",
                transform_membership,
                path=skin_path,
            )

    @staticmethod
    def _rewrite_unified_constraints(output: dict[str, Any]) -> None:
        if "constraints" in output:
            raise ValueError(
                "document.constraints is reserved for the Spine 4.3 codec and cannot "
                "be supplied through document extras"
            )

        ordered: list[tuple[int, dict[str, Any]]] = []
        for collection_name in ("ik", "transform"):
            raw_collection = output.pop(collection_name, None)
            if raw_collection is None:
                continue
            collection = _require_list(
                raw_collection,
                path=f"document.{collection_name}",
            )
            for index, raw_constraint in enumerate(collection):
                constraint_path = f"document.{collection_name}[{index}]"
                constraint = _require_dict(raw_constraint, path=constraint_path)
                copied = dict(constraint)
                _require_name(copied.get("name"), path=f"{constraint_path}.name")
                order = _require_order(
                    copied.pop("order", 0),
                    path=f"{constraint_path}.order",
                )
                if collection_name == "ik":
                    copied["type"] = "ik"
                else:
                    Spine43JsonCodec._rewrite_transform_constraint(
                        copied,
                        path=constraint_path,
                    )
                ordered.append((order, copied))

        orders = tuple(order for order, _constraint in ordered)
        if len(set(orders)) != len(orders):
            raise ValueError(
                "Spine 4.3 unified constraint order must be globally unique: "
                f"{orders}"
            )
        expected_orders = tuple(range(len(ordered)))
        if tuple(sorted(orders)) != expected_orders:
            raise ValueError(
                "Spine 4.3 unified constraint order must be contiguous 0..N-1: "
                f"actual={orders}, expected={expected_orders}"
            )

        names = tuple(constraint["name"] for _order, constraint in ordered)
        if len(set(names)) != len(names):
            raise ValueError(
                "Spine 4.3 unified constraint names must be globally unique: "
                f"{names}"
            )

        if ordered:
            output["constraints"] = [
                constraint
                for _order, constraint in sorted(ordered, key=lambda item: item[0])
            ]

    @staticmethod
    def _rewrite_transform_constraint(
        constraint: dict[str, Any],
        *,
        path: str,
    ) -> None:
        target = _require_name(
            constraint.pop("target", None),
            path=f"{path}.target",
        )
        constraint["type"] = "transform"
        constraint["source"] = target

        local = constraint.pop("local", None)
        if local is not None:
            if not isinstance(local, bool):
                raise TypeError(f"{path}.local must be bool")
            if local:
                constraint["localSource"] = True
                constraint["localTarget"] = True

        relative = constraint.pop("relative", None)
        if relative is not None:
            if not isinstance(relative, bool):
                raise TypeError(f"{path}.relative must be bool")
            if relative:
                constraint["additive"] = True

        if "properties" in constraint:
            raise ValueError(
                f"{path}.properties is reserved for Spine 4.3 codec mapping"
            )
        constraint["properties"] = _same_property_mapping()

    @staticmethod
    def _reject_untranslated_constraint_families(output: dict[str, Any]) -> None:
        unsupported = tuple(
            field_name
            for field_name in ("path", "physics", "slider")
            if field_name in output
        )
        if unsupported:
            raise ValueError(
                "Spine 4.3 codec cannot translate untyped root constraint families: "
                f"{unsupported}"
            )


__all__ = ["Spine43JsonCodec"]
