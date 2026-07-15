"""Stable A1 fingerprints for comparing legacy and rewritten Spine outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Tuple


@dataclass(frozen=True, slots=True)
class LegacyCompatibilityFingerprint:
    """Ordered structural signature of a Spine export.

    Volatile skeleton metadata and numeric mesh coordinates are intentionally not
    included. Geometry parity is checked separately; this fingerprint protects the
    A1 rig/slot/attachment/constraint contract during the rewrite.
    """

    bone_names: Tuple[str, ...]
    bone_parents: Tuple[str | None, ...]
    slot_pairs: Tuple[Tuple[str, str], ...]
    ik_entries: Tuple[Tuple[str, int, Tuple[str, ...], str], ...]
    transform_entries: Tuple[Tuple[str, int, Tuple[str, ...], str], ...]
    skin_names: Tuple[str, ...]
    attachment_paths: Tuple[Tuple[str, str, str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def build_legacy_fingerprint(data: Mapping[str, Any]) -> LegacyCompatibilityFingerprint:
    """Create a deterministic A1 fingerprint from an exported Spine JSON mapping."""

    document = _require_mapping(data, "data")
    bones_raw = document.get("bones", [])
    slots_raw = document.get("slots", [])
    skins_raw = document.get("skins", [])
    ik_raw = document.get("ik", [])
    transform_raw = document.get("transform", [])

    if not all(isinstance(collection, list) for collection in (
        bones_raw, slots_raw, skins_raw, ik_raw, transform_raw
    )):
        raise TypeError("bones, slots, skins, ik and transform must be lists")

    bones = tuple(_require_mapping(item, f"bones[{index}]") for index, item in enumerate(bones_raw))
    slots = tuple(_require_mapping(item, f"slots[{index}]") for index, item in enumerate(slots_raw))
    ik = tuple(_require_mapping(item, f"ik[{index}]") for index, item in enumerate(ik_raw))
    transform = tuple(
        _require_mapping(item, f"transform[{index}]")
        for index, item in enumerate(transform_raw)
    )

    attachment_paths: list[tuple[str, str, str, str]] = []
    skin_names: list[str] = []
    for skin_index, skin_value in enumerate(skins_raw):
        skin = _require_mapping(skin_value, f"skins[{skin_index}]")
        skin_name = str(skin.get("name", "default"))
        skin_names.append(skin_name)
        attachments = _require_mapping(
            skin.get("attachments", {}),
            f"skins[{skin_index}].attachments",
        )
        for slot_name, slot_value in attachments.items():
            slot_attachments = _require_mapping(
                slot_value,
                f"skins[{skin_index}].attachments.{slot_name}",
            )
            for attachment_name, attachment_value in slot_attachments.items():
                attachment = _require_mapping(
                    attachment_value,
                    f"skins[{skin_index}].attachments.{slot_name}.{attachment_name}",
                )
                attachment_paths.append(
                    (
                        skin_name,
                        str(slot_name),
                        str(attachment_name),
                        str(attachment.get("type", "region")),
                    )
                )

    return LegacyCompatibilityFingerprint(
        bone_names=tuple(str(item["name"]) for item in bones),
        bone_parents=tuple(
            None if item.get("parent") is None else str(item["parent"])
            for item in bones
        ),
        slot_pairs=tuple((str(item["name"]), str(item["bone"])) for item in slots),
        ik_entries=tuple(
            (
                str(item["name"]),
                int(item.get("order", 0)),
                tuple(str(name) for name in item.get("bones", [])),
                str(item["target"]),
            )
            for item in ik
        ),
        transform_entries=tuple(
            (
                str(item["name"]),
                int(item.get("order", 0)),
                tuple(str(name) for name in item.get("bones", [])),
                str(item["target"]),
            )
            for item in transform
        ),
        skin_names=tuple(skin_names),
        attachment_paths=tuple(attachment_paths),
    )
