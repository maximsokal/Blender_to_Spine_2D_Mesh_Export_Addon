"""Native Spine 4.1 JSON codec for the Rewrite canonical document subset."""

from __future__ import annotations

import json
from typing import Any

from ..model import SpineDocument
from ..serializer import SpineSerializer
from ..version_target import SpineJsonTarget
from .base import SpineJsonCodecContext, SpineJsonVersionCodec


_CONSTRAINT_COLLECTIONS = ("ik", "transform", "path")


def _require_dict(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{path} must be a JSON object")
    return value


def _require_list(value: Any, *, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{path} must be a JSON array")
    return value


def _require_constraint_order(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{path} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _extend_unique_names(
    mapping: dict[str, Any],
    key: str,
    names: list[str],
    *,
    path: str,
) -> None:
    if not names:
        return

    existing_value = mapping.get(key)
    if existing_value is None:
        mapping[key] = list(names)
        return

    existing = _require_list(existing_value, path=f"{path}.{key}")
    if not all(isinstance(name, str) and name for name in existing):
        raise TypeError(f"{path}.{key} must contain non-empty strings")

    for name in names:
        if name not in existing:
            existing.append(name)


def _constraint_records(
    output: dict[str, Any],
) -> list[tuple[int, int, str, int, dict[str, Any]]]:
    """Collect every Spine 4.1 constraint with a deterministic stable tie-break."""

    records: list[tuple[int, int, str, int, dict[str, Any]]] = []
    encounter_index = 0
    for collection_name in _CONSTRAINT_COLLECTIONS:
        raw_collection = output.get(collection_name)
        if raw_collection is None:
            continue
        constraints = _require_list(
            raw_collection,
            path=f"document.{collection_name}",
        )
        for item_index, value in enumerate(constraints):
            constraint_path = f"document.{collection_name}[{item_index}]"
            constraint = _require_dict(value, path=constraint_path)
            original_order = _require_constraint_order(
                constraint.get("order", 0),
                path=f"{constraint_path}.order",
            )
            records.append(
                (
                    original_order,
                    encounter_index,
                    collection_name,
                    item_index,
                    constraint,
                )
            )
            encounter_index += 1
    return records


def _normalize_constraint_orders(output: dict[str, Any]) -> None:
    """Linearize tied or gapped canonical orders for the Spine 4.1 loader.

    Connected composition intentionally allows constraints on one Z layer to share an
    order. Spine 4.1 imports require one global ordinal for every IK, transform, and
    path constraint. Sorting by authored order preserves phase semantics; the stable
    encounter index preserves deterministic component order within a tie.
    """

    records = _constraint_records(output)
    ordered = sorted(records, key=lambda item: (item[0], item[1]))
    for normalized_order, record in enumerate(ordered):
        record[4]["order"] = normalized_order

    final_orders = sorted(
        _require_constraint_order(
            constraint.get("order"),
            path=f"document.{collection_name}[{item_index}].order",
        )
        for (
            _original_order,
            _encounter_index,
            collection_name,
            item_index,
            constraint,
        ) in records
    )
    expected_orders = list(range(len(records)))
    if final_orders != expected_orders:
        raise ValueError(
            "Spine 4.1 constraint order normalization failed: "
            f"expected={expected_orders}, actual={final_orders}"
        )


class Spine41JsonCodec(SpineJsonVersionCodec):
    """Encode the add-on's canonical setup-pose subset as Spine 4.1.24 JSON."""

    @property
    def target(self) -> SpineJsonTarget:
        return SpineJsonTarget.SPINE_4_1

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
                f"Spine41JsonCodec requires {self.target.value}, "
                f"got {context.target.value}"
            )

        # Serialize once through the proven canonical validator/encoder. Parsing the
        # resulting JSON gives this codec a fully detached mapping, so every downgrade
        # operation is guaranteed not to mutate SpineDocument or nested animation data.
        canonical_json = SpineSerializer(validator=context.validator).to_json(
            document,
            indent=indent,
        )
        output = _require_dict(json.loads(canonical_json), path="document")

        self._rewrite_skeleton(output)
        self._rewrite_bone_transform_fields(output)
        self._rewrite_skin_constraint_membership(output, document)
        self._remove_physics(output)
        _normalize_constraint_orders(output)

        return json.dumps(
            output,
            ensure_ascii=False,
            indent=indent,
            allow_nan=False,
        )

    def _rewrite_skeleton(self, output: dict[str, Any]) -> None:
        skeleton = _require_dict(output.get("skeleton"), path="document.skeleton")
        skeleton["spine"] = self.target.exact_version
        # referenceScale is read by the 4.2 runtime but is not part of the 4.1 loader.
        skeleton.pop("referenceScale", None)

    @staticmethod
    def _rewrite_bone_transform_fields(output: dict[str, Any]) -> None:
        bones = _require_list(output.get("bones"), path="document.bones")
        for index, value in enumerate(bones):
            bone = _require_dict(value, path=f"document.bones[{index}]")
            if "inherit" not in bone:
                continue
            if "transform" in bone:
                raise ValueError(
                    f"document.bones[{index}] contains both inherit and transform"
                )
            bone["transform"] = bone.pop("inherit")

    @staticmethod
    def _rewrite_skin_constraint_membership(
        output: dict[str, Any],
        document: SpineDocument,
    ) -> None:
        ik_names = {constraint.name for constraint in document.ik}
        transform_names = {constraint.name for constraint in document.transform}
        ambiguous_names = ik_names.intersection(transform_names)
        if ambiguous_names:
            raise ValueError(
                "IK and transform constraint names must be globally unique for "
                f"Spine 4.1 skin membership: {tuple(sorted(ambiguous_names))}"
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
            for item_index, name in enumerate(constraints):
                if not isinstance(name, str) or not name:
                    raise TypeError(
                        f"{skin_path}.constraints[{item_index}] must be a "
                        "non-empty string"
                    )
                if name in ik_names:
                    ik_membership.append(name)
                elif name in transform_names:
                    transform_membership.append(name)
                else:
                    raise ValueError(
                        f"{skin_path}.constraints references unsupported or unknown "
                        f"constraint {name!r} for Spine 4.1"
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
    def _remove_physics(output: dict[str, Any]) -> None:
        # Spine 4.1 has no physics constraints or physics animation timelines. The
        # current add-on does not generate them, but raw document extras may contain
        # them, so downgrade deterministically instead of leaking 4.2-only sections.
        output.pop("physics", None)

        animations = output.get("animations")
        if animations is None:
            return
        animations_mapping = _require_dict(animations, path="document.animations")
        for animation_name, value in animations_mapping.items():
            animation = _require_dict(
                value,
                path=f"document.animations.{animation_name}",
            )
            animation.pop("physics", None)


__all__ = ["Spine41JsonCodec"]
