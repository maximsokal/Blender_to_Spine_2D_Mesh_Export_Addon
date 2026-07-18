"""Typed static-audit findings, module records, and reviewed suppressions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Mapping


SUPPRESSION_PATTERN = re.compile(r"#\s*pipeline-audit:\s*ignore=([A-Za-z0-9_,*-]+)")


class AuditSeverity(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True, slots=True)
class AuditFinding:
    severity: AuditSeverity
    code: str
    message: str
    line: int
    function: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.severity, AuditSeverity):
            raise TypeError("severity must be AuditSeverity")
        if not isinstance(self.code, str) or not self.code.strip():
            raise ValueError("code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message.strip():
            raise ValueError("message must be a non-empty string")
        if not isinstance(self.line, int) or self.line < 1:
            raise ValueError("line must be a positive integer")


@dataclass(frozen=True, slots=True)
class ModuleAudit:
    module: str
    relative_path: str
    layer: str
    line_count: int
    function_count: int
    class_count: int
    internal_imports: tuple[str, ...]
    findings: tuple[AuditFinding, ...]

    @property
    def score(self) -> int:
        weights = {AuditSeverity.ERROR: 10, AuditSeverity.WARNING: 3, AuditSeverity.INFO: 1}
        return sum(weights[item.severity] for item in self.findings)


def suppression_map(source: str) -> dict[int, frozenset[str]]:
    result: dict[int, frozenset[str]] = {}
    for line_number, line in enumerate(source.splitlines(), start=1):
        match = SUPPRESSION_PATTERN.search(line)
        if match is None:
            continue
        codes = frozenset(
            value.strip().upper() for value in match.group(1).split(",") if value.strip()
        )
        if codes:
            result[line_number] = codes
    return result


def is_suppressed(
    finding: AuditFinding,
    suppressions: Mapping[int, frozenset[str]],
) -> bool:
    for line_number in (finding.line, finding.line - 1):
        codes = suppressions.get(line_number, frozenset())
        if "*" in codes or "ALL" in codes or finding.code.upper() in codes:
            return True
    return False


def finding_payload(item: AuditFinding) -> dict[str, Any]:
    return {
        "severity": item.severity.value,
        "code": item.code,
        "message": item.message,
        "line": item.line,
        "function": item.function,
    }


def module_payload(item: ModuleAudit) -> dict[str, Any]:
    return {
        "module": item.module,
        "relative_path": item.relative_path,
        "layer": item.layer,
        "line_count": item.line_count,
        "function_count": item.function_count,
        "class_count": item.class_count,
        "internal_imports": list(item.internal_imports),
        "score": item.score,
        "findings": [finding_payload(finding) for finding in item.findings],
    }


__all__ = [
    "AuditFinding", "AuditSeverity", "ModuleAudit", "finding_payload",
    "is_suppressed", "module_payload", "suppression_map",
]
