from __future__ import annotations

import ast
import logging
import pathlib
import shutil
import sys
import tempfile
from dataclasses import dataclass
from typing import Union

import libcst as cst
import libcst.matchers as m
from libcst.codemod import CodemodContext, VisitorBasedCodemodCommand
from libcst.metadata import MetadataWrapper, ParentNodeProvider


MODULE_NAME = "Blender_to_Spine2D_Mesh_Exporter"

REQUIRES = ["black", "isort", "fake-bpy-module-4.1", "libcst"]

CLEANER_SKIP = {}

INCLUDE_ONLY = {
    "uv_operations.py",
    "utils.py",
    "ui.py",
    "texture_baker_integration.py",
    "texture_baker.py",
    "seam_marker.py",
    "plane_cut.py",
    "multi_object_export.py",
    "main.py",
    "json_merger.py",
    "json_export.py",
    "config.py",
    "__init__.py",
}
ADDITIONAL_FILES = {"blender_manifest.toml"}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("prepare_package")


@dataclass(slots=True)
class Args:
    src: pathlib.Path
    out: pathlib.Path


def parse(argv: list[str]) -> Args:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--src", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    ns = p.parse_args(argv)
    return Args(ns.src.resolve(), ns.out.resolve())


def patch_logging_levels(source: str) -> str:
    import re

    source = re.sub(r'("level"\s*:\s*")[A-Z]+(" )', r"\1ERROR\2", source, flags=re.I)
    source = re.sub(
        r"(basicConfig\s*\([^)]*level\s*=\s*logging\.)[A-Z]+",
        r"\1ERROR",
        source,
        flags=re.I,
    )
    return source


def run_formatters(py_file: pathlib.Path) -> None:
    import subprocess, sys, logging

    for tool in ("isort", "black"):
        cmd = [sys.executable, "-m", tool, str(py_file), "--quiet"]
        try:
            subprocess.check_call(cmd)
        except FileNotFoundError:
            logging.warning("[%s] not installed; skipping %s", tool, py_file.name)
        except subprocess.CalledProcessError as exc:
            logging.error("[%s] failed on %s: %s", tool, py_file.name, exc)


class BlenderCSTCleaner(VisitorBasedCodemodCommand):
    METADATA_DEPENDENCIES = (ParentNodeProvider,)

    def __init__(self, context: CodemodContext):
        super().__init__(context)
        self.is_in_blender_class = False
        self.is_in_register_func = False

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        blender_types = (
            "Operator",
            "Panel",
            "PropertyGroup",
            "AddonPreferences",
            "Menu",
        )
        for base in node.bases:
            if m.matches(
                base.value,
                m.Attribute(
                    value=m.Attribute(value=m.Name("bpy"), attr=m.Name("types"))
                ),
            ):
                if (
                    isinstance(base.value, cst.Attribute)
                    and base.value.attr.value in blender_types
                ):
                    self.is_in_blender_class = True
                    break
        return True

    def leave_ClassDef(
        self, original_node: cst.ClassDef, updated_node: cst.ClassDef
    ) -> cst.ClassDef:
        self.is_in_blender_class = False
        return updated_node

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if node.name.value in ("register", "unregister"):
            self.is_in_register_func = True
        return True

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        self.is_in_register_func = False
        return updated_node

    @m.leave(m.SimpleStatementLine(body=[m.Expr(value=m.SimpleString())]))
    def filter_docstring(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> Union[cst.BaseStatement, cst.RemovalSentinel]:
        parent = self.get_metadata(ParentNodeProvider, original_node)
        is_module_level = isinstance(parent, cst.Module)

        if self.is_in_blender_class or self.is_in_register_func or is_module_level:
            return updated_node

        if isinstance(parent, cst.IndentedBlock):
            grandparent = self.get_metadata(ParentNodeProvider, parent)
            if isinstance(grandparent, (cst.FunctionDef, cst.ClassDef)):
                if grandparent.body.body and grandparent.body.body[0] == original_node:
                    if self.is_in_blender_class or self.is_in_register_func:
                        return updated_node

        return cst.RemoveFromParent()

    @m.leave(m.Assert())
    def remove_asserts(
        self, original_node: cst.Assert, updated_node: cst.Assert
    ) -> cst.RemovalSentinel:
        return cst.RemoveFromParent()

    @m.leave(
        m.Expr(
            value=m.Call(
                func=m.OneOf(m.Name("print"), m.Attribute(attr=m.Name("debug")))
            )
        )
    )
    def remove_debug_and_print_calls(
        self, original_node: cst.Expr, updated_node: cst.Expr
    ) -> Union[cst.Expr, cst.RemovalSentinel]:
        return cst.RemoveFromParent()

    @m.leave(m.EmptyLine(comment=m.Comment()))
    def remove_comment_lines(
        self, original_node: cst.EmptyLine, updated_node: cst.EmptyLine
    ) -> cst.RemovalSentinel:
        return cst.RemoveFromParent()


def _validate_structure(source: str, filename: str):
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as e:
        log.error(f"AST parsing failed for {filename}: {e}")
        raise

    has_register = any(
        isinstance(node, ast.FunctionDef) and node.name == "register"
        for node in ast.walk(tree)
    )
    has_unregister = any(
        isinstance(node, ast.FunctionDef) and node.name == "unregister"
        for node in ast.walk(tree)
    )

    if not has_register and not has_unregister:
        if filename != "__init__.py":
            log.warning(
                f"Structural warning in {filename}: No register() or unregister() function found."
            )

    blender_classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if (
                    isinstance(base, ast.Attribute)
                    and isinstance(base.value, ast.Attribute)
                    and isinstance(base.value.value, ast.Name)
                    and base.value.value.id == "bpy"
                    and base.value.attr == "types"
                ):
                    blender_classes.append(node.name)

    if not blender_classes and filename not in (
        "utils.py",
        "config.py",
        "__init__.py",
    ):
        log.warning(
            f"Structural warning in {filename}: No Blender-specific classes (Operator, Panel, etc.) found."
        )


def process_file(src_path: pathlib.Path, dst_path: pathlib.Path) -> None:
    log.info(f"Processing: {src_path.name}")
    run_formatters(src_path)
    raw_source = src_path.read_text(encoding="utf-8")

    log.info(f"1. Validating original source: {src_path.name}")
    try:
        compile(raw_source, f"<raw> {src_path.name}", "exec")
        _validate_structure(raw_source, src_path.name)
        log.info("   ✅ Original source is valid.")
    except Exception as exc:
        log.error(f"   ❌ Initial validation FAILED for {src_path.name}: {exc}")
        raise

    cleaned_source = raw_source
    if src_path.name not in CLEANER_SKIP:
        log.info(f"2. Cleaning with BlenderCSTCleaner: {src_path.name}")
        try:
            tree = cst.parse_module(raw_source)
            wrapper = MetadataWrapper(tree)
            context = cst.codemod.CodemodContext()
            cleaner = BlenderCSTCleaner(context)
            modified_tree = wrapper.visit(cleaner)
            cleaned_source = modified_tree.code
            log.info("   ✅ Successfully cleaned.")
        except Exception as e:
            log.warning(
                f"   ⚠️ LibCST failed on {src_path.name}: {e}. Using raw source."
            )
            cleaned_source = raw_source
    else:
        log.info(f"2. Skipping cleaning for {src_path.name}.")

    final_source = patch_logging_levels(cleaned_source)

    log.info(f"3. Validating final source: {src_path.name}")
    try:
        compile(final_source, src_path.name, "exec")
        _validate_structure(final_source, src_path.name)
        log.info("   ✅ Final source is valid.")
    except Exception as exc:
        log.error(f"   ❌ Final validation FAILED for {src_path.name}: {exc}")
        raise

    dst_path.write_text(final_source, encoding="utf-8")


def main():
    log.info("--- Running full cleanup before build ---")
    project_root = pathlib.Path(__file__).parent.parent
    for folder_name in ("build", "dist"):
        folder_path = project_root / folder_name
        if folder_path.exists():
            log.info(f"Removing folder: {folder_path}")
            shutil.rmtree(folder_path, ignore_errors=True)
    temp_dir = pathlib.Path(tempfile.gettempdir())
    for item in temp_dir.iterdir():
        if item.is_dir() and item.name.startswith("pkg_tmp_"):
            log.info(f"Removing temp build folder: {item}")
            shutil.rmtree(item, ignore_errors=True)
    log.info("--- Cleanup finished ---")

    args = parse(sys.argv[1:])
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="pkg_tmp_"))

    module_source_dir = args.src / MODULE_NAME
    if not module_source_dir.is_dir():
        log.error(f"Source module directory not found: {module_source_dir}")
        sys.exit(1)

    try:
        target_dir_in_tmp = tmp / MODULE_NAME
        target_dir_in_tmp.mkdir()
        log.info(f"Created temporary subdirectory: {target_dir_in_tmp}")

        log.info(
            f"Copying source files from {module_source_dir} to {target_dir_in_tmp}"
        )
        files_to_copy = INCLUDE_ONLY | ADDITIONAL_FILES
        copied_py_files = []

        for filename in files_to_copy:
            src_file = module_source_dir / filename
            if src_file.exists():
                dest_file = target_dir_in_tmp / filename
                shutil.copy2(src_file, dest_file)
                if dest_file.suffix == ".py":
                    copied_py_files.append(dest_file)
            else:
                log.warning(f"Source file not found, skipping: {src_file}")

        for py_file in copied_py_files:
            process_file(py_file, py_file)

        log.info(f"Creating archive: {args.out}")
        archive_base_name = args.out.with_suffix("")

        shutil.make_archive(
            base_name=str(archive_base_name),
            format="zip",
            root_dir=tmp,
            base_dir=MODULE_NAME,
        )

        log.info(f"Successfully created {args.out}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        log.info("Build finished.")


if __name__ == "__main__":
    main()
