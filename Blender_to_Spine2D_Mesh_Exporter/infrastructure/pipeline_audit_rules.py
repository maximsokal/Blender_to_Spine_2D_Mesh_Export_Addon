"""AST visitor implementing Blender and Rewrite architecture audit rules."""

from __future__ import annotations

import ast
from collections import Counter
from typing import Any, Mapping

from .pipeline_audit_model import AuditFinding, AuditSeverity


PRODUCTION_LAYERS = frozenset({"application", "domain", "blender_adapter", "infrastructure"})
ALLOWED_IMPORT_LAYERS: Mapping[str, frozenset[str]] = {
    "domain": frozenset({"domain"}),
    "application": frozenset({"application", "domain"}),
    "infrastructure": frozenset({"infrastructure"}),
    "blender_adapter": frozenset({"blender_adapter", "application", "domain", "infrastructure"}),
    "root": frozenset({"root", "blender_adapter", "application", "domain", "infrastructure"}),
}
DIRECT_FILE_METHODS = frozenset({"write_text", "write_bytes", "unlink", "rename", "touch"})

class PipelineAuditVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        module: str,
        package_context: str,
        layer: str,
        package_name: str,
    ) -> None:
        self.module = module
        self.package_context = package_context
        self.layer = layer
        self.package_name = package_name
        self.parents: dict[ast.AST, ast.AST] = {}
        self.function_stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        self.findings: list[AuditFinding] = []
        self.internal_imports: set[str] = set()
        self.function_count = 0
        self.class_count = 0

    def visit(self, node: ast.AST) -> Any:
        for child in ast.iter_child_nodes(node):
            self.parents[child] = node
        return super().visit(node)

    def _function_name(self) -> str | None:
        return None if not self.function_stack else self.function_stack[-1].name

    def _finding(
        self,
        severity: AuditSeverity,
        code: str,
        message: str,
        node: ast.AST,
        *,
        function: str | None = None,
    ) -> None:
        self.findings.append(
            AuditFinding(
                severity=severity,
                code=code,
                message=message,
                line=max(1, int(getattr(node, "lineno", 1))),
                function=self._function_name() if function is None else function,
            )
        )

    def _ancestor(self, node: ast.AST, kinds: tuple[type[ast.AST], ...]) -> ast.AST | None:
        current = self.parents.get(node)
        while current is not None:
            if isinstance(current, kinds):
                return current
            current = self.parents.get(current)
        return None

    def _resolve_import_from(self, node: ast.ImportFrom) -> str | None:
        if node.level == 0:
            return node.module
        package_parts = (
            [] if not self.package_context else self.package_context.split(".")
        )
        remove_count = max(0, node.level - 1)
        if remove_count > len(package_parts):
            return node.module
        base = package_parts[: len(package_parts) - remove_count]
        if node.module:
            base.extend(node.module.split("."))
        return ".".join(base)

    def _internal_relative_name(self, imported: str | None) -> str | None:
        if not imported:
            return None
        if imported == self.package_name:
            return ""
        prefix = self.package_name + "."
        if imported.startswith(prefix):
            return imported[len(prefix) :]
        first = imported.split(".", 1)[0]
        if first in PRODUCTION_LAYERS:
            return imported
        return None

    @staticmethod
    def _layer_from_module(module: str) -> str:
        first = module.split(".", 1)[0] if module else "root"
        return first if first in PRODUCTION_LAYERS else "root"

    def _check_internal_import(self, imported: str, node: ast.AST) -> None:
        relative = self._internal_relative_name(imported)
        if relative is None:
            return
        self.internal_imports.add(relative or "<package>")
        imported_layer = self._layer_from_module(relative)
        allowed = ALLOWED_IMPORT_LAYERS.get(self.layer, ALLOWED_IMPORT_LAYERS["root"])
        if imported_layer not in allowed:
            self._finding(
                AuditSeverity.ERROR,
                "LAYER_IMPORT_VIOLATION",
                f"{self.layer} module imports forbidden {imported_layer} layer: {imported}",
                node,
            )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_internal_import(alias.name, node)
            root = alias.name.split(".", 1)[0]
            if root in {"bpy", "bmesh"} and self.layer in {
                "domain",
                "application",
                "infrastructure",
            }:
                self._finding(
                    AuditSeverity.ERROR,
                    "BLENDER_IMPORT_OUTSIDE_ADAPTER",
                    f"{self.layer} layer imports {root}",
                    node,
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        imported = self._resolve_import_from(node)
        if imported:
            self._check_internal_import(imported, node)
            root = imported.split(".", 1)[0]
            if root in {"bpy", "bmesh"} and self.layer in {
                "domain",
                "application",
                "infrastructure",
            }:
                self._finding(
                    AuditSeverity.ERROR,
                    "BLENDER_IMPORT_OUTSIDE_ADAPTER",
                    f"{self.layer} layer imports {root}",
                    node,
                )
        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.function_count += 1
        end_line = int(getattr(node, "end_lineno", node.lineno))
        length = end_line - int(node.lineno) + 1
        if length > 180:
            self._finding(
                AuditSeverity.WARNING,
                "OVERSIZED_FUNCTION",
                f"Function contains {length} lines; split by pipeline responsibility",
                node,
                function=node.name,
            )
        self.function_stack.append(node)
        self.generic_visit(node)
        self._check_bmesh_lifetime(node)
        self.function_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_count += 1
        self.generic_visit(node)

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        target = node.func
        if isinstance(target, ast.Name):
            return target.id
        parts: list[str] = []
        current: ast.AST = target
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))

    def visit_Call(self, node: ast.Call) -> None:
        name = self._call_name(node)
        if name.startswith("bpy.ops.") and self._ancestor(
            node,
            (ast.For, ast.AsyncFor, ast.While, ast.comprehension),
        ):
            self._finding(
                AuditSeverity.WARNING,
                "BPY_OPS_IN_LOOP",
                f"Operator {name} is called inside a loop",
                node,
            )
        if name == "print" and self.layer in PRODUCTION_LAYERS:
            self._finding(
                AuditSeverity.WARNING,
                "PRINT_IN_PRODUCTION",
                "Use the per-file logger instead of print()",
                node,
            )
        if self.layer != "infrastructure":
            if name == "open" or name in {"os.replace", "shutil.copy", "shutil.copy2"}:
                self._finding(
                    AuditSeverity.WARNING,
                    "DIRECT_FILE_IO_OUTSIDE_INFRASTRUCTURE",
                    f"Direct filesystem mutation through {name}",
                    node,
                )
            if isinstance(node.func, ast.Attribute) and node.func.attr in DIRECT_FILE_METHODS:
                self._finding(
                    AuditSeverity.WARNING,
                    "DIRECT_FILE_IO_OUTSIDE_INFRASTRUCTURE",
                    f"Direct filesystem mutation through .{node.func.attr}()",
                    node,
                )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self._finding(
                AuditSeverity.ERROR,
                "BARE_EXCEPTION",
                "Bare except catches BaseException and can hide cancellation/system exits",
                node,
            )
        broad = False
        if isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}:
            broad = True
        if isinstance(node.type, ast.Tuple):
            broad = any(
                isinstance(item, ast.Name) and item.id in {"Exception", "BaseException"}
                for item in node.type.elts
            )

        has_raise = any(isinstance(item, ast.Raise) for item in ast.walk(node))
        call_names = {
            self._call_name(item)
            for item in ast.walk(node)
            if isinstance(item, ast.Call)
        }
        has_log = any(
            name.endswith((".debug", ".info", ".warning", ".error", ".exception", ".critical"))
            for name in call_names
        )
        has_structured_failure = any(
            name.endswith(("build_a1_failure_result", "build_multi_object_failure_result"))
            for name in call_names
        )
        pass_only = bool(node.body) and all(isinstance(item, ast.Pass) for item in node.body)
        if pass_only and not has_raise and not has_log:
            self._finding(
                AuditSeverity.ERROR,
                "SILENT_EXCEPTION",
                "Exception handler suppresses failure without logging or re-raising",
                node,
            )
        elif broad and not has_raise and not has_log and not has_structured_failure:
            has_return = any(isinstance(item, ast.Return) for item in ast.walk(node))
            self._finding(
                AuditSeverity.INFO if has_return else AuditSeverity.WARNING,
                "BROAD_EXCEPTION_FALLBACK" if has_return else "BROAD_EXCEPTION_WITHOUT_VISIBILITY",
                (
                    "Broad exception returns an explicit fallback value"
                    if has_return
                    else "Broad exception is handled without logger or re-raise"
                ),
                node,
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self.function_stack and isinstance(node.value, (ast.List, ast.Dict, ast.Set)):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and not target.id.isupper()
                    and not target.id.startswith("__")
                ):
                    self._finding(
                        AuditSeverity.INFO,
                        "MUTABLE_MODULE_GLOBAL",
                        f"Mutable module global '{target.id}' can leak state between exports",
                        node,
                    )
        self.generic_visit(node)

    def _check_bmesh_lifetime(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        allocations: dict[str, ast.Call] = {}
        free_calls: Counter[str] = Counter()
        final_free_names: set[str] = set()
        parent_map = {child: parent for parent in ast.walk(function) for child in ast.iter_child_nodes(parent)}

        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                if self._call_name(node.value) != "bmesh.new":
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        allocations[target.id] = node.value
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "free" or not isinstance(node.func.value, ast.Name):
                continue
            name = node.func.value.id
            free_calls[name] += 1
            current: ast.AST | None = node
            while current is not None:
                parent = parent_map.get(current)
                if isinstance(parent, ast.Try) and current in parent.finalbody:
                    final_free_names.add(name)
                    break
                current = parent

        for name, allocation in allocations.items():
            count = free_calls[name]
            if count == 0:
                self._finding(
                    AuditSeverity.ERROR,
                    "BMESH_FREE_MISSING",
                    f"bmesh.new() assigned to '{name}' is never freed",
                    allocation,
                    function=function.name,
                )
            elif name not in final_free_names:
                self._finding(
                    AuditSeverity.WARNING,
                    "BMESH_FREE_NOT_GUARANTEED",
                    f"'{name}.free()' is not guaranteed by a finally block",
                    allocation,
                    function=function.name,
                )
            if count > 1:
                self._finding(
                    AuditSeverity.WARNING,
                    "BMESH_DOUBLE_FREE_RISK",
                    f"'{name}.free()' appears {count} times",
                    allocation,
                    function=function.name,
                )



__all__ = ["PipelineAuditVisitor", "PRODUCTION_LAYERS"]
