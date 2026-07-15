"""Validation of source lineage propagated through evaluated Blender modifiers."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Tuple


class ModifierLineagePolicy(str, Enum):
    """How much topology change an evaluated modifier stack may introduce."""

    STRICT_PRESERVE = "STRICT_PRESERVE"
    ALLOW_SOURCE_DUPLICATION = "ALLOW_SOURCE_DUPLICATION"


class LineageSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"


@dataclass(frozen=True, slots=True)
class LineageIssue:
    severity: LineageSeverity
    code: str
    channel: str
    message: str


@dataclass(frozen=True, slots=True)
class LineageChannelReport:
    channel: str
    source_count: int
    evaluated_count: int
    unknown_count: int
    out_of_range_values: Tuple[int, ...]
    missing_source_indices: Tuple[int, ...]
    duplicated_source_indices: Tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EvaluatedLineageReport:
    policy: ModifierLineagePolicy
    vertices: LineageChannelReport
    edges: LineageChannelReport
    faces: LineageChannelReport
    corners: LineageChannelReport
    issues: Tuple[LineageIssue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity is LineageSeverity.ERROR for issue in self.issues)


class EvaluatedLineageError(ValueError):
    def __init__(self, report: EvaluatedLineageReport):
        self.report = report
        message = "Evaluated modifier lineage is unsafe:\n" + "\n".join(
            f"- [{issue.code}] {issue.channel}: {issue.message}"
            for issue in report.issues
            if issue.severity is LineageSeverity.ERROR
        )
        super().__init__(message)


def _validate_source_count(value: int, field_name: str) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")


def _analyse_channel(
    channel: str,
    values: Iterable[int | None],
    source_count: int,
) -> LineageChannelReport:
    _validate_source_count(source_count, f"{channel}.source_count")
    resolved = tuple(values)
    known = tuple(value for value in resolved if value is not None)
    for value in known:
        if not isinstance(value, int):
            raise TypeError(f"{channel} lineage values must be int or None")

    out_of_range = tuple(
        sorted({value for value in known if value < 0 or value >= source_count})
    )
    valid_known = tuple(value for value in known if 0 <= value < source_count)
    counts = Counter(valid_known)
    missing = tuple(index for index in range(source_count) if counts[index] == 0)
    duplicated = tuple(sorted(index for index, count in counts.items() if count > 1))
    return LineageChannelReport(
        channel=channel,
        source_count=source_count,
        evaluated_count=len(resolved),
        unknown_count=sum(value is None for value in resolved),
        out_of_range_values=out_of_range,
        missing_source_indices=missing,
        duplicated_source_indices=duplicated,
    )


def _corner_identity_values(
    source_face_indices: Iterable[int | None],
    source_corner_indices: Iterable[int | None],
    source_face_corner_counts: Tuple[int, ...],
) -> tuple[tuple[int | None, ...], list[LineageIssue]]:
    face_values = tuple(source_face_indices)
    corner_values = tuple(source_corner_indices)
    if len(face_values) != len(corner_values):
        raise ValueError("corner face and corner index arrays must have equal length")

    identities: list[int | None] = []
    issues: list[LineageIssue] = []
    offsets: list[int] = []
    total = 0
    for face_index, corner_count in enumerate(source_face_corner_counts):
        _validate_source_count(corner_count, f"source_face_corner_counts[{face_index}]")
        offsets.append(total)
        total += corner_count

    for evaluated_index, (face_index, corner_index) in enumerate(
        zip(face_values, corner_values)
    ):
        if face_index is None or corner_index is None:
            identities.append(None)
            continue
        if not isinstance(face_index, int) or not isinstance(corner_index, int):
            raise TypeError("corner lineage values must be int or None")
        if face_index < 0 or face_index >= len(source_face_corner_counts):
            identities.append(total + max(0, face_index))
            issues.append(
                LineageIssue(
                    LineageSeverity.ERROR,
                    "CORNER_FACE_OUT_OF_RANGE",
                    "corners",
                    f"Evaluated corner {evaluated_index} references source face {face_index}",
                )
            )
            continue
        corner_count = source_face_corner_counts[face_index]
        if corner_index < 0 or corner_index >= corner_count:
            identities.append(total + corner_index)
            issues.append(
                LineageIssue(
                    LineageSeverity.ERROR,
                    "CORNER_INDEX_OUT_OF_RANGE",
                    "corners",
                    f"Evaluated corner {evaluated_index} references corner {corner_index} "
                    f"of source face {face_index} with {corner_count} corners",
                )
            )
            continue
        identities.append(offsets[face_index] + corner_index)
    return tuple(identities), issues


def analyse_evaluated_lineage(
    *,
    source_vertex_count: int,
    source_edge_count: int,
    source_face_corner_counts: Tuple[int, ...],
    vertex_source_indices: Iterable[int | None],
    edge_source_indices: Iterable[int | None],
    face_source_indices: Iterable[int | None],
    corner_source_face_indices: Iterable[int | None],
    corner_source_corner_indices: Iterable[int | None],
    policy: ModifierLineagePolicy = ModifierLineagePolicy.STRICT_PRESERVE,
) -> EvaluatedLineageReport:
    """Validate decoded lineage attributes from one evaluated mesh."""

    if not isinstance(policy, ModifierLineagePolicy):
        raise TypeError("policy must be ModifierLineagePolicy")
    _validate_source_count(source_vertex_count, "source_vertex_count")
    _validate_source_count(source_edge_count, "source_edge_count")
    if not isinstance(source_face_corner_counts, tuple):
        raise TypeError("source_face_corner_counts must be tuple")

    corner_identities, corner_issues = _corner_identity_values(
        corner_source_face_indices,
        corner_source_corner_indices,
        source_face_corner_counts,
    )
    vertex_report = _analyse_channel(
        "vertices", vertex_source_indices, source_vertex_count
    )
    edge_report = _analyse_channel("edges", edge_source_indices, source_edge_count)
    face_report = _analyse_channel(
        "faces", face_source_indices, len(source_face_corner_counts)
    )
    corner_report = _analyse_channel(
        "corners", corner_identities, sum(source_face_corner_counts)
    )

    issues: list[LineageIssue] = list(corner_issues)
    required_reports = (vertex_report, face_report, corner_report)
    for report in required_reports:
        if report.unknown_count:
            issues.append(
                LineageIssue(
                    LineageSeverity.ERROR,
                    "UNKNOWN_SOURCE_LINEAGE",
                    report.channel,
                    f"{report.unknown_count} evaluated elements have no source lineage",
                )
            )
        if report.out_of_range_values:
            issues.append(
                LineageIssue(
                    LineageSeverity.ERROR,
                    "SOURCE_INDEX_OUT_OF_RANGE",
                    report.channel,
                    f"Out-of-range source indices: {report.out_of_range_values}",
                )
            )

    # New evaluated edges are allowed because MeshEdge explicitly supports
    # ``source_id=None``. Vertices, faces, and corners cannot be invented because
    # their identity is required for stable UV and Spine correspondence.
    if edge_report.out_of_range_values:
        issues.append(
            LineageIssue(
                LineageSeverity.ERROR,
                "SOURCE_INDEX_OUT_OF_RANGE",
                "edges",
                f"Out-of-range source indices: {edge_report.out_of_range_values}",
            )
        )
    if edge_report.unknown_count:
        issues.append(
            LineageIssue(
                LineageSeverity.WARNING,
                "GENERATED_EDGES",
                "edges",
                f"{edge_report.unknown_count} evaluated edges are generated",
            )
        )

    if policy is ModifierLineagePolicy.STRICT_PRESERVE:
        for report in (vertex_report, edge_report, face_report, corner_report):
            if report.evaluated_count != report.source_count:
                issues.append(
                    LineageIssue(
                        LineageSeverity.ERROR,
                        "TOPOLOGY_COUNT_CHANGED",
                        report.channel,
                        f"Source count {report.source_count}, evaluated count "
                        f"{report.evaluated_count}",
                    )
                )
            if report.missing_source_indices:
                issues.append(
                    LineageIssue(
                        LineageSeverity.ERROR,
                        "SOURCE_ELEMENTS_MISSING",
                        report.channel,
                        f"Missing source indices: {report.missing_source_indices}",
                    )
                )
            if report.duplicated_source_indices:
                issues.append(
                    LineageIssue(
                        LineageSeverity.ERROR,
                        "SOURCE_ELEMENTS_DUPLICATED",
                        report.channel,
                        f"Duplicated source indices: {report.duplicated_source_indices}",
                    )
                )

    return EvaluatedLineageReport(
        policy=policy,
        vertices=vertex_report,
        edges=edge_report,
        faces=face_report,
        corners=corner_report,
        issues=tuple(issues),
    )


def require_valid_evaluated_lineage(report: EvaluatedLineageReport) -> None:
    if not isinstance(report, EvaluatedLineageReport):
        raise TypeError("report must be EvaluatedLineageReport")
    if not report.valid:
        raise EvaluatedLineageError(report)
