"""Cross-reference and structural validation for generated Spine documents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Iterable, Tuple

from .model import MeshAttachment, SpineDocument
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


def _duplicates(values: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates


class SpineValidator:
    """Validate A1 compatibility invariants before JSON serialization."""

    def validate(self, document: SpineDocument) -> Tuple[SpineValidationIssue, ...]:
        if not isinstance(document, SpineDocument):
            raise TypeError("document must be SpineDocument")

        issues: list[SpineValidationIssue] = []
        bone_names = [bone.name for bone in document.bones]
        bone_name_set = set(bone_names)
        slot_names = [slot.name for slot in document.slots]

        for duplicate in sorted(_duplicates(bone_names)):
            issues.append(
                SpineValidationIssue(
                    ValidationSeverity.ERROR,
                    "DUPLICATE_BONE",
                    "bones",
                    f"Bone name '{duplicate}' is duplicated",
                )
            )
        for duplicate in sorted(_duplicates(slot_names)):
            issues.append(
                SpineValidationIssue(
                    ValidationSeverity.ERROR,
                    "DUPLICATE_SLOT",
                    "slots",
                    f"Slot name '{duplicate}' is duplicated",
                )
            )

        index_by_bone = {name: index for index, name in enumerate(bone_names)}
        for index, bone in enumerate(document.bones):
            if bone.parent is None:
                continue
            if bone.parent not in bone_name_set:
                issues.append(
                    SpineValidationIssue(
                        ValidationSeverity.ERROR,
                        "MISSING_BONE_PARENT",
                        f"bones[{index}].parent",
                        f"Parent bone '{bone.parent}' does not exist",
                    )
                )
            elif index_by_bone[bone.parent] >= index:
                issues.append(
                    SpineValidationIssue(
                        ValidationSeverity.ERROR,
                        "PARENT_AFTER_CHILD",
                        f"bones[{index}]",
                        f"Parent '{bone.parent}' must appear before child '{bone.name}'",
                    )
                )

        for index, slot in enumerate(document.slots):
            if slot.bone not in bone_name_set:
                issues.append(
                    SpineValidationIssue(
                        ValidationSeverity.ERROR,
                        "MISSING_SLOT_BONE",
                        f"slots[{index}].bone",
                        f"Slot '{slot.name}' references missing bone '{slot.bone}'",
                    )
                )

        constraint_names = [item.name for item in document.ik] + [
            item.name for item in document.transform
        ]
        for duplicate in sorted(_duplicates(constraint_names)):
            issues.append(
                SpineValidationIssue(
                    ValidationSeverity.ERROR,
                    "DUPLICATE_CONSTRAINT",
                    "constraints",
                    f"Constraint name '{duplicate}' is duplicated",
                )
            )

        constraint_orders: dict[int, str] = {}
        for collection_name, constraints in (
            ("ik", document.ik),
            ("transform", document.transform),
        ):
            for index, constraint in enumerate(constraints):
                for referenced_bone in constraint.bones:
                    if referenced_bone not in bone_name_set:
                        issues.append(
                            SpineValidationIssue(
                                ValidationSeverity.ERROR,
                                "MISSING_CONSTRAINT_BONE",
                                f"{collection_name}[{index}].bones",
                                f"Constraint '{constraint.name}' references missing bone "
                                f"'{referenced_bone}'",
                            )
                        )
                if constraint.target not in bone_name_set:
                    issues.append(
                        SpineValidationIssue(
                            ValidationSeverity.ERROR,
                            "MISSING_CONSTRAINT_TARGET",
                            f"{collection_name}[{index}].target",
                            f"Constraint '{constraint.name}' references missing target "
                            f"'{constraint.target}'",
                        )
                    )
                previous = constraint_orders.get(constraint.order)
                if previous is not None:
                    issues.append(
                        SpineValidationIssue(
                            ValidationSeverity.ERROR,
                            "DUPLICATE_CONSTRAINT_ORDER",
                            f"{collection_name}[{index}].order",
                            f"Order {constraint.order} is shared by '{previous}' and "
                            f"'{constraint.name}'",
                        )
                    )
                else:
                    constraint_orders[constraint.order] = constraint.name

        slot_name_set = set(slot_names)
        constraint_name_set = set(constraint_names)
        for skin_index, skin in enumerate(document.skins):
            for bone_name in skin.bones:
                if bone_name not in bone_name_set:
                    issues.append(
                        SpineValidationIssue(
                            ValidationSeverity.ERROR,
                            "MISSING_SKIN_BONE",
                            f"skins[{skin_index}].bones",
                            f"Skin '{skin.name}' references missing bone '{bone_name}'",
                        )
                    )
            for constraint_name in skin.constraints:
                if constraint_name not in constraint_name_set:
                    issues.append(
                        SpineValidationIssue(
                            ValidationSeverity.ERROR,
                            "MISSING_SKIN_CONSTRAINT",
                            f"skins[{skin_index}].constraints",
                            f"Skin '{skin.name}' references missing constraint "
                            f"'{constraint_name}'",
                        )
                    )
            for slot_name, attachments in skin.attachments.items():
                if slot_name not in slot_name_set:
                    issues.append(
                        SpineValidationIssue(
                            ValidationSeverity.ERROR,
                            "MISSING_SKIN_SLOT",
                            f"skins[{skin_index}].attachments.{slot_name}",
                            f"Skin '{skin.name}' references missing slot '{slot_name}'",
                        )
                    )
                for attachment_name, attachment in attachments.items():
                    if isinstance(attachment, MeshAttachment):
                        issues.extend(
                            self._validate_mesh_attachment(
                                attachment,
                                path=(
                                    f"skins[{skin_index}].attachments."
                                    f"{slot_name}.{attachment_name}"
                                ),
                                bone_count=len(document.bones),
                            )
                        )

        return tuple(issues)

    def _validate_mesh_attachment(
        self,
        attachment: MeshAttachment,
        *,
        path: str,
        bone_count: int,
    ) -> list[SpineValidationIssue]:
        issues: list[SpineValidationIssue] = []
        vertex_count = len(attachment.uvs) // 2

        if attachment.hull > vertex_count:
            issues.append(
                SpineValidationIssue(
                    ValidationSeverity.ERROR,
                    "INVALID_HULL",
                    f"{path}.hull",
                    f"Hull {attachment.hull} exceeds vertex count {vertex_count}",
                )
            )

        for triangle_index, vertex_index in enumerate(attachment.triangles):
            if not isinstance(vertex_index, int):
                issues.append(
                    SpineValidationIssue(
                        ValidationSeverity.ERROR,
                        "NON_INTEGER_TRIANGLE_INDEX",
                        f"{path}.triangles[{triangle_index}]",
                        "Triangle indices must be integers",
                    )
                )
            elif vertex_index < 0 or vertex_index >= vertex_count:
                issues.append(
                    SpineValidationIssue(
                        ValidationSeverity.ERROR,
                        "TRIANGLE_INDEX_OUT_OF_RANGE",
                        f"{path}.triangles[{triangle_index}]",
                        f"Vertex index {vertex_index} is outside [0, {vertex_count})",
                    )
                )

        for value_index, value in enumerate((*attachment.uvs, *attachment.vertices)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                issues.append(
                    SpineValidationIssue(
                        ValidationSeverity.ERROR,
                        "NON_NUMERIC_MESH_VALUE",
                        f"{path}.numeric[{value_index}]",
                        "Mesh arrays must contain numeric values",
                    )
                )
            elif not isfinite(float(value)):
                issues.append(
                    SpineValidationIssue(
                        ValidationSeverity.ERROR,
                        "NON_FINITE_MESH_VALUE",
                        f"{path}.numeric[{value_index}]",
                        "Mesh arrays cannot contain NaN or infinity",
                    )
                )

        if len(attachment.vertices) == vertex_count * 2:
            return issues

        try:
            weighted_vertices = decode_weighted_vertices(
                attachment.vertices,
                expected_vertex_count=vertex_count,
            )
            for vertex_index, vertex in enumerate(weighted_vertices):
                weight_sum = sum(item.weight for item in vertex.influences)
                if abs(weight_sum - 1.0) > 1e-4:
                    issues.append(
                        SpineValidationIssue(
                            ValidationSeverity.WARNING,
                            "WEIGHTS_NOT_NORMALIZED",
                            f"{path}.vertices[{vertex_index}]",
                            f"Influence weights sum to {weight_sum:.6f}, expected 1.0",
                        )
                    )
                for influence_index, influence in enumerate(vertex.influences):
                    if influence.bone_index >= bone_count:
                        issues.append(
                            SpineValidationIssue(
                                ValidationSeverity.ERROR,
                                "BONE_INDEX_OUT_OF_RANGE",
                                f"{path}.vertices[{vertex_index}]."
                                f"influences[{influence_index}]",
                                f"Bone index {influence.bone_index} exceeds document "
                                f"bone count {bone_count}",
                            )
                        )
        except (TypeError, ValueError) as exc:
            issues.append(
                SpineValidationIssue(
                    ValidationSeverity.ERROR,
                    "INVALID_WEIGHTED_VERTEX_STREAM",
                    f"{path}.vertices",
                    str(exc),
                )
            )
        return issues

    def validate_or_raise(self, document: SpineDocument) -> None:
        issues = self.validate(document)
        errors = [issue for issue in issues if issue.severity is ValidationSeverity.ERROR]
        if errors:
            raise SpineValidationError(errors)
