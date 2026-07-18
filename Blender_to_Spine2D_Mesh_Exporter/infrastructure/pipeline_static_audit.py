"""Static architecture and reliability audit for every production addon file."""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from .pipeline_audit_model import (
    AuditFinding,
    AuditSeverity,
    ModuleAudit,
    finding_payload,
    is_suppressed,
    module_payload,
    suppression_map,
)
from .pipeline_audit_rules import PRODUCTION_LAYERS, PipelineAuditVisitor


SCHEMA_VERSION = 1


def module_name(relative_path: Path) -> str:
    without_suffix = relative_path.with_suffix("")
    parts = without_suffix.parts
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or "<package>"


def layer_for_path(relative_path: Path) -> str:
    if not relative_path.parts:
        return "root"
    first = relative_path.parts[0]
    return first if first in PRODUCTION_LAYERS else "root"


def audit_module_source(
    source: str,
    *,
    module: str,
    relative_path: str,
    layer: str,
    package_name: str,
) -> ModuleAudit:
    """Audit one source string and return deterministic findings."""

    lines = source.splitlines()
    try:
        tree = ast.parse(source, filename=relative_path)
    except SyntaxError as exc:
        finding = AuditFinding(
            severity=AuditSeverity.ERROR,
            code="SYNTAX_ERROR",
            message=exc.msg,
            line=max(1, int(exc.lineno or 1)),
        )
        return ModuleAudit(
            module=module, relative_path=relative_path, layer=layer,
            line_count=len(lines), function_count=0, class_count=0,
            internal_imports=(), findings=(finding,),
        )

    path = Path(relative_path)
    package_context = (
        module if path.name == "__init__.py" and module != "<package>"
        else module.rpartition(".")[0] if module != "<package>"
        else ""
    )
    visitor = PipelineAuditVisitor(
        module=module,
        package_context=package_context,
        layer=layer,
        package_name=package_name,
    )
    visitor.visit(tree)
    findings = list(visitor.findings)
    if len(lines) > 800:
        findings.append(AuditFinding(
            severity=AuditSeverity.WARNING,
            code="OVERSIZED_MODULE",
            message=f"Module contains {len(lines)} lines; split by responsibility",
            line=1,
        ))
    suppressions = suppression_map(source)
    findings = [item for item in findings if not is_suppressed(item, suppressions)]
    findings.sort(key=lambda item: (item.line, item.severity.value, item.code))
    return ModuleAudit(
        module=module, relative_path=relative_path, layer=layer,
        line_count=len(lines), function_count=visitor.function_count,
        class_count=visitor.class_count,
        internal_imports=tuple(sorted(visitor.internal_imports)),
        findings=tuple(findings),
    )


def audit_pipeline_package(
    package_directory: Path,
    *,
    package_name: str,
    focus_modules: Sequence[str] = (),
) -> dict[str, Any]:
    """Audit every production file and return a JSON-compatible package report."""

    if not isinstance(package_directory, Path):
        raise TypeError("package_directory must be pathlib.Path")
    root = package_directory.expanduser().resolve(strict=False)
    if not root.is_dir():
        raise ValueError(f"package_directory is not a directory: {root}")
    if not isinstance(package_name, str) or not package_name.strip():
        raise ValueError("package_name must be a non-empty string")
    focus = tuple(value.casefold() for value in focus_modules if value.strip())

    audits: list[ModuleAudit] = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts:
            continue
        layer = layer_for_path(relative)
        if layer == "root" and len(relative.parts) > 1:
            continue
        module = module_name(relative)
        if focus and not any(
            query in module.casefold() or query in relative.as_posix().casefold()
            for query in focus
        ):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = path.read_text(encoding="utf-8-sig")
        audits.append(audit_module_source(
            source, module=module, relative_path=relative.as_posix(),
            layer=layer, package_name=package_name,
        ))

    severity_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    for audit in audits:
        for finding in audit.findings:
            severity_counts[finding.severity.value] += 1
            code_counts[finding.code] += 1

    weak_spots = sorted(
        (audit for audit in audits if audit.findings),
        key=lambda item: (item.score, item.line_count),
        reverse=True,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "package_name": package_name,
        "package_directory": str(root),
        "focus_queries": list(focus_modules),
        "summary": {
            "module_count": len(audits),
            "finding_count": sum(severity_counts.values()),
            "error_count": severity_counts[AuditSeverity.ERROR.value],
            "warning_count": severity_counts[AuditSeverity.WARNING.value],
            "info_count": severity_counts[AuditSeverity.INFO.value],
        },
        "finding_codes": dict(sorted(code_counts.items())),
        "weak_spots": [{
            "module": item.module,
            "relative_path": item.relative_path,
            "score": item.score,
            "finding_count": len(item.findings),
            "top_findings": [finding_payload(finding) for finding in item.findings[:10]],
        } for item in weak_spots[:50]],
        "modules": [module_payload(item) for item in audits],
    }


__all__ = [
    "AuditFinding", "AuditSeverity", "ModuleAudit",
    "audit_module_source", "audit_pipeline_package",
]
