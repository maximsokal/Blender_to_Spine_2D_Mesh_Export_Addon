"""Structured golden comparison for legacy and rewritten A1 Spine exports.

Byte-for-byte JSON equality is too strict for exporter migration because volatile
metadata and harmless floating-point representation may differ. At the same time,
name/order changes or weighted bone-index corruption must never be hidden. This
module compares stable A1 structure, setup data, mesh topology, UV coordinates, and
decoded weighted-vertex semantics while collecting every mismatch in one report.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fnmatch import fnmatchcase
from math import isclose, isfinite
from typing import Any, Mapping, Sequence, Tuple

from .golden import LegacyCompatibilityFingerprint, build_legacy_fingerprint
from .weighted_vertices import WeightedVertex, decode_weighted_vertices


class A1ParitySeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True, slots=True)
class A1ParityIssue:
    severity: A1ParitySeverity
    code: str
    path: str
    message: str
    expected: Any = None
    actual: Any = None


@dataclass(frozen=True, slots=True)
class A1ParitySettings:
    absolute_tolerance: float = 1e-4
    relative_tolerance: float = 1e-6
    ignored_paths: Tuple[str, ...] = (
        "skeleton.hash",
        "skeleton.images",
        "skeleton.audio",
    )
    compare_animations: bool = False
    nonessential_mesh_edges_are_errors: bool = False

    def __post_init__(self) -> None:
        for field_name in ("absolute_tolerance", "relative_tolerance"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
            if value < 0.0:
                raise ValueError(f"{field_name} cannot be negative")
        if not isinstance(self.ignored_paths, tuple) or not all(
            isinstance(path, str) and path for path in self.ignored_paths
        ):
            raise TypeError("ignored_paths must be a tuple of non-empty strings")
        if not isinstance(self.compare_animations, bool):
            raise TypeError("compare_animations must be bool")
        if not isinstance(self.nonessential_mesh_edges_are_errors, bool):
            raise TypeError("nonessential_mesh_edges_are_errors must be bool")


@dataclass(frozen=True, slots=True)
class A1ParityReport:
    settings: A1ParitySettings
    expected_fingerprint: LegacyCompatibilityFingerprint | None
    actual_fingerprint: LegacyCompatibilityFingerprint | None
    issues: Tuple[A1ParityIssue, ...]

    @property
    def compatible(self) -> bool:
        return not any(
            issue.severity is A1ParitySeverity.ERROR for issue in self.issues
        )

    @property
    def error_count(self) -> int:
        return sum(
            issue.severity is A1ParitySeverity.ERROR for issue in self.issues
        )

    @property
    def warning_count(self) -> int:
        return sum(
            issue.severity is A1ParitySeverity.WARNING for issue in self.issues
        )

    def require_compatible(self) -> None:
        if self.compatible:
            return
        details = "\n".join(
            f"- [{issue.code}] {issue.path}: {issue.message}"
            for issue in self.issues
            if issue.severity is A1ParitySeverity.ERROR
        )
        raise A1ParityError("A1 golden parity failed:\n" + details, self)


class A1ParityError(ValueError):
    def __init__(self, message: str, report: A1ParityReport):
        self.report = report
        super().__init__(message)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _path_ignored(path: str, settings: A1ParitySettings) -> bool:
    return any(fnmatchcase(path, pattern) for pattern in settings.ignored_paths)


def _append_issue(
    issues: list[A1ParityIssue],
    *,
    severity: A1ParitySeverity,
    code: str,
    path: str,
    message: str,
    expected: Any = None,
    actual: Any = None,
) -> None:
    issues.append(
        A1ParityIssue(
            severity=severity,
            code=code,
            path=path,
            message=message,
            expected=expected,
            actual=actual,
        )
    )


def _compare_value(
    expected: Any,
    actual: Any,
    *,
    path: str,
    settings: A1ParitySettings,
    issues: list[A1ParityIssue],
    severity: A1ParitySeverity = A1ParitySeverity.ERROR,
) -> None:
    """Recursively compare JSON-compatible values with explicit numeric tolerance."""

    if _path_ignored(path, settings):
        return

    if _is_number(expected) and _is_number(actual):
        if not isclose(
            float(expected),
            float(actual),
            rel_tol=settings.relative_tolerance,
            abs_tol=settings.absolute_tolerance,
        ):
            _append_issue(
                issues,
                severity=severity,
                code="NUMERIC_MISMATCH",
                path=path,
                message=(
                    f"Expected {expected!r}, got {actual!r} outside configured "
                    "numeric tolerance"
                ),
                expected=expected,
                actual=actual,
            )
        return

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys, key=str):
            child_path = f"{path}.{key}" if path else str(key)
            if _path_ignored(child_path, settings):
                continue
            _append_issue(
                issues,
                severity=severity,
                code="MISSING_FIELD",
                path=child_path,
                message="Field is missing from rewritten output",
                expected=expected[key],
            )
        for key in sorted(actual_keys - expected_keys, key=str):
            child_path = f"{path}.{key}" if path else str(key)
            if _path_ignored(child_path, settings):
                continue
            _append_issue(
                issues,
                severity=severity,
                code="UNEXPECTED_FIELD",
                path=child_path,
                message="Rewritten output contains an unexpected field",
                actual=actual[key],
            )
        for key in sorted(expected_keys & actual_keys, key=str):
            child_path = f"{path}.{key}" if path else str(key)
            _compare_value(
                expected[key],
                actual[key],
                path=child_path,
                settings=settings,
                issues=issues,
                severity=severity,
            )
        return

    if (
        isinstance(expected, Sequence)
        and not isinstance(expected, (str, bytes))
        and isinstance(actual, Sequence)
        and not isinstance(actual, (str, bytes))
    ):
        if len(expected) != len(actual):
            _append_issue(
                issues,
                severity=severity,
                code="LENGTH_MISMATCH",
                path=path,
                message=f"Expected {len(expected)} entries, got {len(actual)}",
                expected=len(expected),
                actual=len(actual),
            )
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual)
        ):
            _compare_value(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                settings=settings,
                issues=issues,
                severity=severity,
            )
        return

    if type(expected) is not type(actual) and expected != actual:
        _append_issue(
            issues,
            severity=severity,
            code="TYPE_MISMATCH",
            path=path,
            message=(
                f"Expected type {type(expected).__name__}, got "
                f"{type(actual).__name__}"
            ),
            expected=expected,
            actual=actual,
        )
        return

    if expected != actual:
        _append_issue(
            issues,
            severity=severity,
            code="VALUE_MISMATCH",
            path=path,
            message=f"Expected {expected!r}, got {actual!r}",
            expected=expected,
            actual=actual,
        )


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping")
    return value


def _require_sequence(value: Any, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{path} must be a sequence")
    return value


def _attachment_map(
    document: Mapping[str, Any],
) -> dict[Tuple[str, str, str], Mapping[str, Any]]:
    result: dict[Tuple[str, str, str], Mapping[str, Any]] = {}
    skins = _require_sequence(document.get("skins", ()), "skins")
    for skin_index, skin_value in enumerate(skins):
        skin = _require_mapping(skin_value, f"skins[{skin_index}]")
        skin_name = str(skin.get("name", "default"))
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
                key = (skin_name, str(slot_name), str(attachment_name))
                if key in result:
                    raise ValueError(f"Duplicate attachment path {key}")
                result[key] = _require_mapping(
                    attachment_value,
                    (
                        f"skins[{skin_index}].attachments.{slot_name}."
                        f"{attachment_name}"
                    ),
                )
    return result


def _compare_fingerprints(
    expected: LegacyCompatibilityFingerprint,
    actual: LegacyCompatibilityFingerprint,
    *,
    settings: A1ParitySettings,
    issues: list[A1ParityIssue],
) -> None:
    """Compare stable A1 categories with one deterministic issue per category.

    Structural issue paths are deliberately aggregate and stable. The complete
    ordered tuples are attached to the issue, so callers can display or diff them
    without depending on which element happened to differ first.
    """

    if not isinstance(settings, A1ParitySettings):
        raise TypeError("settings must be A1ParitySettings")
    for field_name in (
        "bone_names",
        "bone_parents",
        "slot_pairs",
        "ik_entries",
        "transform_entries",
        "skin_names",
        "attachment_paths",
    ):
        expected_value = getattr(expected, field_name)
        actual_value = getattr(actual, field_name)
        path = f"structure.{field_name}"
        if _path_ignored(path, settings) or expected_value == actual_value:
            continue
        _append_issue(
            issues,
            severity=A1ParitySeverity.ERROR,
            code="STRUCTURE_MISMATCH",
            path=path,
            message=f"Ordered A1 structural field '{field_name}' differs",
            expected=expected_value,
            actual=actual_value,
        )


def _compare_ordered_section_extras(
    expected_document: Mapping[str, Any],
    actual_document: Mapping[str, Any],
    section_name: str,
    structural_fields: set[str],
    *,
    settings: A1ParitySettings,
    issues: list[A1ParityIssue],
) -> None:
    expected_values = _require_sequence(
        expected_document.get(section_name, ()),
        section_name,
    )
    actual_values = _require_sequence(
        actual_document.get(section_name, ()),
        section_name,
    )
    if len(expected_values) != len(actual_values):
        return
    for index, (expected_value, actual_value) in enumerate(
        zip(expected_values, actual_values)
    ):
        expected_mapping = _require_mapping(
            expected_value,
            f"{section_name}[{index}]",
        )
        actual_mapping = _require_mapping(
            actual_value,
            f"{section_name}[{index}]",
        )
        expected_extras = {
            key: value
            for key, value in expected_mapping.items()
            if key not in structural_fields
        }
        actual_extras = {
            key: value
            for key, value in actual_mapping.items()
            if key not in structural_fields
        }
        _compare_value(
            expected_extras,
            actual_extras,
            path=f"{section_name}[{index}]",
            settings=settings,
            issues=issues,
        )


def _compare_weighted_vertex(
    expected: WeightedVertex,
    actual: WeightedVertex,
    *,
    path: str,
    settings: A1ParitySettings,
    issues: list[A1ParityIssue],
) -> None:
    if len(expected.influences) != len(actual.influences):
        _append_issue(
            issues,
            severity=A1ParitySeverity.ERROR,
            code="INFLUENCE_COUNT_MISMATCH",
            path=path,
            message=(
                f"Expected {len(expected.influences)} influences, got "
                f"{len(actual.influences)}"
            ),
            expected=len(expected.influences),
            actual=len(actual.influences),
        )
        return

    for influence_index, (expected_influence, actual_influence) in enumerate(
        zip(expected.influences, actual.influences)
    ):
        influence_path = f"{path}.influences[{influence_index}]"
        if expected_influence.bone_index != actual_influence.bone_index:
            _append_issue(
                issues,
                severity=A1ParitySeverity.ERROR,
                code="WEIGHTED_BONE_INDEX_MISMATCH",
                path=f"{influence_path}.bone_index",
                message=(
                    f"Expected bone {expected_influence.bone_index}, got "
                    f"{actual_influence.bone_index}"
                ),
                expected=expected_influence.bone_index,
                actual=actual_influence.bone_index,
            )
        for field_name in ("x", "y", "weight"):
            _compare_value(
                getattr(expected_influence, field_name),
                getattr(actual_influence, field_name),
                path=f"{influence_path}.{field_name}",
                settings=settings,
                issues=issues,
            )


def _compare_weighted_vertices(
    expected_stream: Any,
    actual_stream: Any,
    vertex_count: int,
    *,
    path: str,
    settings: A1ParitySettings,
    issues: list[A1ParityIssue],
) -> None:
    try:
        expected_vertices = decode_weighted_vertices(
            expected_stream,
            expected_vertex_count=vertex_count,
        )
    except Exception as exc:
        _append_issue(
            issues,
            severity=A1ParitySeverity.ERROR,
            code="EXPECTED_WEIGHT_STREAM_INVALID",
            path=path,
            message=f"Legacy weighted stream is invalid: {exc}",
            expected=expected_stream,
        )
        return

    try:
        actual_vertices = decode_weighted_vertices(
            actual_stream,
            expected_vertex_count=vertex_count,
        )
    except Exception as exc:
        _append_issue(
            issues,
            severity=A1ParitySeverity.ERROR,
            code="ACTUAL_WEIGHT_STREAM_INVALID",
            path=path,
            message=f"Rewritten weighted stream is invalid: {exc}",
            actual=actual_stream,
        )
        return

    for vertex_index, (expected_vertex, actual_vertex) in enumerate(
        zip(expected_vertices, actual_vertices)
    ):
        _compare_weighted_vertex(
            expected_vertex,
            actual_vertex,
            path=f"{path}[vertex={vertex_index}]",
            settings=settings,
            issues=issues,
        )


def _compare_mesh_attachment(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    path: str,
    settings: A1ParitySettings,
    issues: list[A1ParityIssue],
) -> None:
    expected_uvs = _require_sequence(expected.get("uvs", ()), f"{path}.uvs")
    actual_uvs = _require_sequence(actual.get("uvs", ()), f"{path}.uvs")
    _compare_value(
        expected_uvs,
        actual_uvs,
        path=f"{path}.uvs",
        settings=settings,
        issues=issues,
    )
    expected_vertex_count = len(expected_uvs) // 2
    actual_vertex_count = len(actual_uvs) // 2

    if len(expected_uvs) % 2 != 0:
        _append_issue(
            issues,
            severity=A1ParitySeverity.ERROR,
            code="EXPECTED_UV_STREAM_INVALID",
            path=f"{path}.uvs",
            message="Legacy UV stream length is not divisible by two",
        )
    if len(actual_uvs) % 2 != 0:
        _append_issue(
            issues,
            severity=A1ParitySeverity.ERROR,
            code="ACTUAL_UV_STREAM_INVALID",
            path=f"{path}.uvs",
            message="Rewritten UV stream length is not divisible by two",
        )

    for field_name, default in (
        ("type", "region"),
        ("path", None),
        ("width", 0),
        ("height", 0),
        ("hull", 0),
        ("sequence", None),
        ("triangles", ()),
    ):
        _compare_value(
            expected.get(field_name, default),
            actual.get(field_name, default),
            path=f"{path}.{field_name}",
            settings=settings,
            issues=issues,
        )

    edge_severity = (
        A1ParitySeverity.ERROR
        if settings.nonessential_mesh_edges_are_errors
        else A1ParitySeverity.WARNING
    )
    _compare_value(
        expected.get("edges", ()),
        actual.get("edges", ()),
        path=f"{path}.edges",
        settings=settings,
        issues=issues,
        severity=edge_severity,
    )

    if expected_vertex_count == actual_vertex_count:
        _compare_weighted_vertices(
            expected.get("vertices", ()),
            actual.get("vertices", ()),
            expected_vertex_count,
            path=f"{path}.vertices",
            settings=settings,
            issues=issues,
        )

    excluded = {
        "type",
        "path",
        "width",
        "height",
        "hull",
        "sequence",
        "uvs",
        "triangles",
        "edges",
        "vertices",
    }
    _compare_value(
        {key: value for key, value in expected.items() if key not in excluded},
        {key: value for key, value in actual.items() if key not in excluded},
        path=path,
        settings=settings,
        issues=issues,
    )


def _compare_attachments(
    expected_document: Mapping[str, Any],
    actual_document: Mapping[str, Any],
    *,
    settings: A1ParitySettings,
    issues: list[A1ParityIssue],
) -> None:
    expected_attachments = _attachment_map(expected_document)
    actual_attachments = _attachment_map(actual_document)
    expected_keys = set(expected_attachments)
    actual_keys = set(actual_attachments)

    for key in sorted(expected_keys - actual_keys):
        _append_issue(
            issues,
            severity=A1ParitySeverity.ERROR,
            code="MISSING_ATTACHMENT",
            path="attachments." + "/".join(key),
            message="Attachment is missing from rewritten output",
            expected=key,
        )
    for key in sorted(actual_keys - expected_keys):
        _append_issue(
            issues,
            severity=A1ParitySeverity.ERROR,
            code="UNEXPECTED_ATTACHMENT",
            path="attachments." + "/".join(key),
            message="Rewritten output contains an unexpected attachment",
            actual=key,
        )

    for key in sorted(expected_keys & actual_keys):
        path = "attachments." + "/".join(key)
        expected_attachment = expected_attachments[key]
        actual_attachment = actual_attachments[key]
        expected_type = str(expected_attachment.get("type", "region"))
        actual_type = str(actual_attachment.get("type", "region"))
        if expected_type == "mesh" and actual_type == "mesh":
            _compare_mesh_attachment(
                expected_attachment,
                actual_attachment,
                path=path,
                settings=settings,
                issues=issues,
            )
        else:
            _compare_value(
                expected_attachment,
                actual_attachment,
                path=path,
                settings=settings,
                issues=issues,
            )


def _append_section_error(
    issues: list[A1ParityIssue],
    *,
    section: str,
    side: str,
    exc: Exception,
) -> None:
    _append_issue(
        issues,
        severity=A1ParitySeverity.ERROR,
        code=f"{side.upper()}_{section.upper()}_INVALID",
        path=section,
        message=f"{side.capitalize()} {section} data is invalid: {exc}",
    )


def compare_a1_exports(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    settings: A1ParitySettings | None = None,
) -> A1ParityReport:
    """Compare legacy and rewritten Spine JSON mappings without failing fast."""

    if not isinstance(expected, Mapping):
        raise TypeError("expected must be a mapping")
    if not isinstance(actual, Mapping):
        raise TypeError("actual must be a mapping")
    resolved_settings = settings or A1ParitySettings()
    if not isinstance(resolved_settings, A1ParitySettings):
        raise TypeError("settings must be A1ParitySettings")

    issues: list[A1ParityIssue] = []
    expected_fingerprint: LegacyCompatibilityFingerprint | None = None
    actual_fingerprint: LegacyCompatibilityFingerprint | None = None

    try:
        expected_fingerprint = build_legacy_fingerprint(expected)
    except Exception as exc:
        _append_issue(
            issues,
            severity=A1ParitySeverity.ERROR,
            code="EXPECTED_STRUCTURE_INVALID",
            path="structure",
            message=f"Legacy document structure is invalid: {exc}",
        )
    try:
        actual_fingerprint = build_legacy_fingerprint(actual)
    except Exception as exc:
        _append_issue(
            issues,
            severity=A1ParitySeverity.ERROR,
            code="ACTUAL_STRUCTURE_INVALID",
            path="structure",
            message=f"Rewritten document structure is invalid: {exc}",
        )
    if expected_fingerprint is not None and actual_fingerprint is not None:
        _compare_fingerprints(
            expected_fingerprint,
            actual_fingerprint,
            settings=resolved_settings,
            issues=issues,
        )

    try:
        expected_skeleton = _require_mapping(
            expected.get("skeleton", {}),
            "expected.skeleton",
        )
    except Exception as exc:
        _append_section_error(
            issues,
            section="skeleton",
            side="expected",
            exc=exc,
        )
        expected_skeleton = {}
    try:
        actual_skeleton = _require_mapping(
            actual.get("skeleton", {}),
            "actual.skeleton",
        )
    except Exception as exc:
        _append_section_error(
            issues,
            section="skeleton",
            side="actual",
            exc=exc,
        )
        actual_skeleton = {}
    _compare_value(
        expected_skeleton,
        actual_skeleton,
        path="skeleton",
        settings=resolved_settings,
        issues=issues,
    )

    for section_name, structural_fields in (
        ("bones", {"name", "parent"}),
        ("slots", {"name", "bone"}),
        ("ik", {"name", "order", "bones", "target"}),
        ("transform", {"name", "order", "bones", "target"}),
    ):
        try:
            _compare_ordered_section_extras(
                expected,
                actual,
                section_name,
                structural_fields,
                settings=resolved_settings,
                issues=issues,
            )
        except Exception as exc:
            _append_issue(
                issues,
                severity=A1ParitySeverity.ERROR,
                code="ORDERED_SECTION_INVALID",
                path=section_name,
                message=f"Unable to compare ordered section: {exc}",
            )

    try:
        _compare_attachments(
            expected,
            actual,
            settings=resolved_settings,
            issues=issues,
        )
    except Exception as exc:
        _append_issue(
            issues,
            severity=A1ParitySeverity.ERROR,
            code="ATTACHMENT_SECTION_INVALID",
            path="attachments",
            message=f"Unable to compare attachments: {exc}",
        )

    if resolved_settings.compare_animations:
        _compare_value(
            expected.get("animations", {}),
            actual.get("animations", {}),
            path="animations",
            settings=resolved_settings,
            issues=issues,
        )

    return A1ParityReport(
        settings=resolved_settings,
        expected_fingerprint=expected_fingerprint,
        actual_fingerprint=actual_fingerprint,
        issues=tuple(issues),
    )
