"""Cross-reference and structural validation for generated Spine documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Iterable, Tuple

from .model import MeshAttachment, SpineDocument
from .spine_json_contract import SpineJsonContractError, validate_json_mapping
from .weighted_vertices import decode_weighted_vertices


class ValidationSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True, slots=True)
class SpineValidationIssue:
    severity: ValidationSeverity
    code: str
    path: str
    message: str


class SpineValidationError(ValueError):
    """Raised when ``validate_or_raise`` finds one or more errors."""

    def __init__(self, issues: Iterable[SpineValidationIssue]):
        self.issues = tuple(issues)
        message = "Spine document validation failed:\n" + "\n".join(
            f"- [{issue.code}] {issue.path}: {issue.message}" for issue in self.issues
        )
        super().__init__(message)


def _issue(
    code: str,
    path: str,
    message: str,
    *,
    severity: ValidationSeverity = ValidationSeverity.ERROR,
) -> SpineValidationIssue:
    return SpineValidationIssue(severity, code, path, message)


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(value, int) or isfinite(value)


class SpineValidator:
    """Validate A1 compatibility and strict JSON invariants before serialization."""

    def validate(self, document: SpineDocument) -> Tuple[SpineValidationIssue, ...]:
        if not isinstance(document, SpineDocument):
            raise TypeError("document must be SpineDocument")

        issues: list[SpineValidationIssue] = []
        for field_name in ("skeleton", "animations", "events"):
            issues.extend(
                self._validate_json_mapping(
                    getattr(document, field_name),
                    path=field_name,
                )
            )
        issues.extend(
            self._validate_json_mapping(document.extras, path="document.extras")
        )

        bone_names = tuple(bone.name for bone in document.bones)
        slot_names = tuple(slot.name for slot in document.slots)
        skin_names = tuple(skin.name for skin in document.skins)
        for code, path, label, names in (
            ("DUPLICATE_BONE", "bones", "Bone", bone_names),
            ("DUPLICATE_SLOT", "slots", "Slot", slot_names),
            ("DUPLICATE_SKIN", "skins", "Skin", skin_names),
        ):
            for duplicate in sorted(_duplicates(names)):
                issues.append(
                    _issue(code, path, f"{label} name '{duplicate}' is duplicated")
                )

        bone_name_set = set(bone_names)
        issues.extend(self._validate_bones(document, bone_names, bone_name_set))
        issues.extend(self._validate_slots(document, bone_name_set))

        constraint_names = tuple(
            item.name for item in (*document.ik, *document.transform)
        )
        for duplicate in sorted(_duplicates(constraint_names)):
            issues.append(
                _issue(
                    "DUPLICATE_CONSTRAINT",
                    "constraints",
                    f"Constraint name '{duplicate}' is duplicated",
                )
            )
        issues.extend(
            self._validate_constraints(
                document,
                bone_name_set=bone_name_set,
            )
        )

        skin_issues, available_attachments = self._validate_skins(
            document,
            slot_name_set=set(slot_names),
            bone_name_set=bone_name_set,
            constraint_name_set=set(constraint_names),
        )
        issues.extend(skin_issues)
        issues.extend(
            self._validate_setup_attachments(document, available_attachments)
        )
        return tuple(issues)

    def _validate_bones(
        self,
        document: SpineDocument,
        bone_names: Tuple[str, ...],
        bone_name_set: set[str],
    ) -> list[SpineValidationIssue]:
        issues: list[SpineValidationIssue] = []
        index_by_bone = {name: index for index, name in enumerate(bone_names)}
        for index, bone in enumerate(document.bones):
            issues.extend(
                self._validate_json_mapping(
                    bone.extras,
                    path=f"bones[{index}].extras",
                )
            )
            for field_name in (
                "length",
                "x",
                "y",
                "rotation",
                "scale_x",
                "scale_y",
            ):
                value = getattr(bone, field_name)
                if value is not None and not _is_finite_number(value):
                    issues.append(
                        _issue(
                            "INVALID_BONE_NUMBER",
                            f"bones[{index}].{field_name}",
                            "Bone numeric fields must contain finite numbers",
                        )
                    )
            if bone.parent is None:
                continue
            if bone.parent not in bone_name_set:
                issues.append(
                    _issue(
                        "MISSING_BONE_PARENT",
                        f"bones[{index}].parent",
                        f"Parent bone '{bone.parent}' does not exist",
                    )
                )
            elif index_by_bone[bone.parent] >= index:
                issues.append(
                    _issue(
                        "PARENT_AFTER_CHILD",
                        f"bones[{index}]",
                        f"Parent '{bone.parent}' must appear before child "
                        f"'{bone.name}'",
                    )
                )
        return issues

    def _validate_slots(
        self,
        document: SpineDocument,
        bone_name_set: set[str],
    ) -> list[SpineValidationIssue]:
        issues: list[SpineValidationIssue] = []
        for index, slot in enumerate(document.slots):
            issues.extend(
                self._validate_json_mapping(
                    slot.extras,
                    path=f"slots[{index}].extras",
                )
            )
            if slot.bone not in bone_name_set:
                issues.append(
                    _issue(
                        "MISSING_SLOT_BONE",
                        f"slots[{index}].bone",
                        f"Slot '{slot.name}' references missing bone '{slot.bone}'",
                    )
                )
        return issues

    def _validate_constraints(
        self,
        document: SpineDocument,
        *,
        bone_name_set: set[str],
    ) -> list[SpineValidationIssue]:
        issues: list[SpineValidationIssue] = []
        orders: dict[int, str] = {}
        for collection_name, constraints in (
            ("ik", document.ik),
            ("transform", document.transform),
        ):
            for index, constraint in enumerate(constraints):
                base_path = f"{collection_name}[{index}]"
                issues.extend(
                    self._validate_json_mapping(
                        constraint.extras,
                        path=f"{base_path}.extras",
                    )
                )
                valid_order = (
                    not isinstance(constraint.order, bool)
                    and isinstance(constraint.order, int)
                    and constraint.order >= 0
                )
                if not valid_order:
                    issues.append(
                        _issue(
                            "INVALID_CONSTRAINT_ORDER",
                            f"{base_path}.order",
                            "Constraint order must be a non-negative integer",
                        )
                    )
                for referenced_bone in constraint.bones:
                    if referenced_bone not in bone_name_set:
                        issues.append(
                            _issue(
                                "MISSING_CONSTRAINT_BONE",
                                f"{base_path}.bones",
                                f"Constraint '{constraint.name}' references missing "
                                f"bone '{referenced_bone}'",
                            )
                        )
                if constraint.target not in bone_name_set:
                    issues.append(
                        _issue(
                            "MISSING_CONSTRAINT_TARGET",
                            f"{base_path}.target",
                            f"Constraint '{constraint.name}' references missing target "
                            f"'{constraint.target}'",
                        )
                    )
                if valid_order:
                    previous = orders.get(constraint.order)
                    if previous is None:
                        orders[constraint.order] = constraint.name
                    else:
                        issues.append(
                            _issue(
                                "DUPLICATE_CONSTRAINT_ORDER",
                                f"{base_path}.order",
                                f"Order {constraint.order} is shared by '{previous}' "
                                f"and '{constraint.name}'",
                            )
                        )
        return issues

    def _validate_skins(
        self,
        document: SpineDocument,
        *,
        slot_name_set: set[str],
        bone_name_set: set[str],
        constraint_name_set: set[str],
    ) -> tuple[list[SpineValidationIssue], set[tuple[str, str]]]:
        issues: list[SpineValidationIssue] = []
        available_attachments: set[tuple[str, str]] = set()
        for skin_index, skin in enumerate(document.skins):
            skin_path = f"skins[{skin_index}]"
            issues.extend(
                self._validate_json_mapping(
                    skin.extras,
                    path=f"{skin_path}.extras",
                )
            )
            for bone_name in skin.bones:
                if bone_name not in bone_name_set:
                    issues.append(
                        _issue(
                            "MISSING_SKIN_BONE",
                            f"{skin_path}.bones",
                            f"Skin '{skin.name}' references missing bone '{bone_name}'",
                        )
                    )
            for constraint_name in skin.constraints:
                if constraint_name not in constraint_name_set:
                    issues.append(
                        _issue(
                            "MISSING_SKIN_CONSTRAINT",
                            f"{skin_path}.constraints",
                            f"Skin '{skin.name}' references missing constraint "
                            f"'{constraint_name}'",
                        )
                    )
            for slot_name, attachments in skin.attachments.items():
                slot_path = f"{skin_path}.attachments.{slot_name}"
                if slot_name not in slot_name_set:
                    issues.append(
                        _issue(
                            "MISSING_SKIN_SLOT",
                            slot_path,
                            f"Skin '{skin.name}' references missing slot '{slot_name}'",
                        )
                    )
                for attachment_name, attachment in attachments.items():
                    available_attachments.add((slot_name, attachment_name))
                    attachment_path = f"{slot_path}.{attachment_name}"
                    if isinstance(attachment, MeshAttachment):
                        issues.extend(
                            self._validate_mesh_attachment(
                                attachment,
                                path=attachment_path,
                                bone_count=len(document.bones),
                            )
                        )
                    elif isinstance(attachment, Mapping):
                        issues.extend(
                            self._validate_raw_attachment(
                                attachment,
                                path=attachment_path,
                                bone_count=len(document.bones),
                            )
                        )
                    else:
                        issues.append(
                            _issue(
                                "INVALID_ATTACHMENT_VALUE",
                                attachment_path,
                                "Attachment must be MeshAttachment or a JSON mapping",
                            )
                        )
        return issues, available_attachments

    def _validate_setup_attachments(
        self,
        document: SpineDocument,
        available_attachments: set[tuple[str, str]],
    ) -> list[SpineValidationIssue]:
        issues: list[SpineValidationIssue] = []
        for index, slot in enumerate(document.slots):
            if slot.attachment is None:
                continue
            if (slot.name, slot.attachment) not in available_attachments:
                issues.append(
                    _issue(
                        "MISSING_SETUP_ATTACHMENT",
                        f"slots[{index}].attachment",
                        f"Setup attachment '{slot.attachment}' does not exist for "
                        f"slot '{slot.name}' in any skin",
                    )
                )
        return issues

    def _validate_json_mapping(
        self,
        value: Any,
        *,
        path: str,
    ) -> list[SpineValidationIssue]:
        try:
            validate_json_mapping(value, path=path)
        except SpineJsonContractError as exc:
            return [_issue("INVALID_JSON_VALUE", exc.path, exc.reason)]
        return []

    def _validate_raw_attachment(
        self,
        attachment: Mapping[str, Any],
        *,
        path: str,
        bone_count: int,
    ) -> list[SpineValidationIssue]:
        issues = self._validate_json_mapping(attachment, path=path)
        attachment_type = attachment.get("type", "region")
        if not isinstance(attachment_type, str):
            issues.append(
                _issue(
                    "INVALID_ATTACHMENT_TYPE",
                    f"{path}.type",
                    "Attachment type must be a string",
                )
            )
            return issues
        if attachment_type != "mesh":
            return issues

        missing_fields = tuple(
            field_name
            for field_name in ("uvs", "triangles", "vertices", "hull")
            if field_name not in attachment
        )
        for field_name in missing_fields:
            issues.append(
                _issue(
                    "MISSING_MESH_FIELD",
                    f"{path}.{field_name}",
                    f"Raw mesh attachment is missing required field '{field_name}'",
                )
            )
        if missing_fields:
            return issues

        issues.extend(
            self._validate_mesh_payload(
                uvs=attachment["uvs"],
                triangles=attachment["triangles"],
                vertices=attachment["vertices"],
                hull=attachment["hull"],
                edges=attachment.get("edges", ()),
                width=attachment.get("width"),
                height=attachment.get("height"),
                path=path,
                bone_count=bone_count,
            )
        )
        sequence = attachment.get("sequence")
        if sequence is not None:
            issues.extend(
                self._validate_json_mapping(sequence, path=f"{path}.sequence")
            )
        return issues

    def _validate_mesh_attachment(
        self,
        attachment: MeshAttachment,
        *,
        path: str,
        bone_count: int,
    ) -> list[SpineValidationIssue]:
        issues = self._validate_json_mapping(
            attachment.extras,
            path=f"{path}.extras",
        )
        if attachment.sequence is not None:
            issues.extend(
                self._validate_json_mapping(
                    attachment.sequence,
                    path=f"{path}.sequence",
                )
            )
        issues.extend(
            self._validate_mesh_payload(
                uvs=attachment.uvs,
                triangles=attachment.triangles,
                vertices=attachment.vertices,
                hull=attachment.hull,
                edges=attachment.edges,
                width=attachment.width,
                height=attachment.height,
                path=path,
                bone_count=bone_count,
            )
        )
        return issues

    def _validate_mesh_payload(
        self,
        *,
        uvs: Any,
        triangles: Any,
        vertices: Any,
        hull: Any,
        edges: Any,
        width: Any,
        height: Any,
        path: str,
        bone_count: int,
    ) -> list[SpineValidationIssue]:
        issues: list[SpineValidationIssue] = []
        vertex_count: int | None = None

        if not _is_sequence(uvs):
            issues.append(
                _issue(
                    "INVALID_UV_ARRAY",
                    f"{path}.uvs",
                    "UVs must be a list or tuple",
                )
            )
        else:
            if len(uvs) % 2:
                issues.append(
                    _issue(
                        "ODD_UV_ARRAY",
                        f"{path}.uvs",
                        "UV array length must be even",
                    )
                )
            else:
                vertex_count = len(uvs) // 2
            issues.extend(self._validate_numeric_sequence(uvs, path=f"{path}.uvs"))

        if not _is_sequence(triangles):
            issues.append(
                _issue(
                    "INVALID_TRIANGLE_ARRAY",
                    f"{path}.triangles",
                    "Triangles must be a list or tuple",
                )
            )
        else:
            if len(triangles) % 3:
                issues.append(
                    _issue(
                        "INVALID_TRIANGLE_ARRAY",
                        f"{path}.triangles",
                        "Triangle array length must be divisible by 3",
                    )
                )
            issues.extend(
                self._validate_index_sequence(
                    triangles,
                    path=f"{path}.triangles",
                    vertex_count=vertex_count,
                    non_integer_code="NON_INTEGER_TRIANGLE_INDEX",
                    out_of_range_code="TRIANGLE_INDEX_OUT_OF_RANGE",
                )
            )

        if not _is_sequence(vertices):
            issues.append(
                _issue(
                    "INVALID_VERTEX_ARRAY",
                    f"{path}.vertices",
                    "Vertices must be a list or tuple",
                )
            )
        else:
            numeric_issues = self._validate_numeric_sequence(
                vertices,
                path=f"{path}.vertices",
            )
            issues.extend(numeric_issues)
            if not numeric_issues and vertex_count is not None:
                issues.extend(
                    self._validate_vertex_stream(
                        vertices,
                        vertex_count=vertex_count,
                        bone_count=bone_count,
                        path=f"{path}.vertices",
                    )
                )

        if isinstance(hull, bool) or not isinstance(hull, int) or hull < 0:
            issues.append(
                _issue(
                    "INVALID_HULL",
                    f"{path}.hull",
                    "Hull must be a non-negative integer",
                )
            )
        elif vertex_count is not None and hull > vertex_count:
            issues.append(
                _issue(
                    "INVALID_HULL",
                    f"{path}.hull",
                    f"Hull {hull} exceeds vertex count {vertex_count}",
                )
            )

        if not _is_sequence(edges):
            issues.append(
                _issue(
                    "INVALID_EDGE_ARRAY",
                    f"{path}.edges",
                    "Edges must be a list or tuple",
                )
            )
        else:
            if len(edges) % 2:
                issues.append(
                    _issue(
                        "INVALID_EDGE_ARRAY",
                        f"{path}.edges",
                        "Edge array length must be divisible into pairs",
                    )
                )
            issues.extend(
                self._validate_index_sequence(
                    edges,
                    path=f"{path}.edges",
                    vertex_count=vertex_count,
                    non_integer_code="NON_INTEGER_EDGE_INDEX",
                    out_of_range_code="EDGE_INDEX_OUT_OF_RANGE",
                )
            )

        for field_name, value in (("width", width), ("height", height)):
            if value is not None and not _is_finite_number(value):
                issues.append(
                    _issue(
                        "INVALID_MESH_DIMENSION",
                        f"{path}.{field_name}",
                        "Mesh dimensions must be finite numbers when present",
                    )
                )
        return issues

    def _validate_vertex_stream(
        self,
        vertices: Sequence[Any],
        *,
        vertex_count: int,
        bone_count: int,
        path: str,
    ) -> list[SpineValidationIssue]:
        if len(vertices) == vertex_count * 2:
            return []
        try:
            weighted_vertices = decode_weighted_vertices(
                vertices,
                expected_vertex_count=vertex_count,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            return [_issue("INVALID_WEIGHTED_VERTEX_STREAM", path, str(exc))]

        issues: list[SpineValidationIssue] = []
        for vertex_index, vertex in enumerate(weighted_vertices):
            weight_sum = sum(item.weight for item in vertex.influences)
            if abs(weight_sum - 1.0) > 1e-4:
                issues.append(
                    _issue(
                        "WEIGHTS_NOT_NORMALIZED",
                        f"{path}[{vertex_index}]",
                        f"Influence weights sum to {weight_sum:.6f}, expected 1.0",
                        severity=ValidationSeverity.WARNING,
                    )
                )
            for influence_index, influence in enumerate(vertex.influences):
                if influence.bone_index >= bone_count:
                    issues.append(
                        _issue(
                            "BONE_INDEX_OUT_OF_RANGE",
                            f"{path}[{vertex_index}].influences[{influence_index}]",
                            f"Bone index {influence.bone_index} exceeds document "
                            f"bone count {bone_count}",
                        )
                    )
        return issues

    def _validate_numeric_sequence(
        self,
        values: Sequence[Any],
        *,
        path: str,
    ) -> list[SpineValidationIssue]:
        issues: list[SpineValidationIssue] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                issues.append(
                    _issue(
                        "NON_NUMERIC_MESH_VALUE",
                        f"{path}[{index}]",
                        "Mesh arrays must contain numeric values, not booleans",
                    )
                )
            elif not _is_finite_number(value):
                issues.append(
                    _issue(
                        "NON_FINITE_MESH_VALUE",
                        f"{path}[{index}]",
                        "Mesh arrays cannot contain NaN or infinity",
                    )
                )
        return issues

    def _validate_index_sequence(
        self,
        values: Sequence[Any],
        *,
        path: str,
        vertex_count: int | None,
        non_integer_code: str,
        out_of_range_code: str,
    ) -> list[SpineValidationIssue]:
        issues: list[SpineValidationIssue] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int):
                issues.append(
                    _issue(
                        non_integer_code,
                        f"{path}[{index}]",
                        "Mesh indices must be integers, not booleans",
                    )
                )
            elif value < 0 or (vertex_count is not None and value >= vertex_count):
                message = (
                    f"Vertex index {value} is negative"
                    if vertex_count is None
                    else f"Vertex index {value} is outside [0, {vertex_count})"
                )
                issues.append(
                    _issue(out_of_range_code, f"{path}[{index}]", message)
                )
        return issues

    def validate_or_raise(self, document: SpineDocument) -> None:
        issues = self.validate(document)
        errors = [
            issue for issue in issues if issue.severity is ValidationSeverity.ERROR
        ]
        if errors:
            raise SpineValidationError(errors)
