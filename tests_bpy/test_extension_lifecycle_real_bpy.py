"""Stress the Rewrite extension lifecycle against Blender 5.2 RNA and imports."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import bpy

import Blender_to_Spine2D_Mesh_Exporter as extension


RNA_REGISTRATIONS = (
    *extension.CONFIG_RNA_PROPERTIES,
    *extension.ui.RNA_PROPERTIES,
    *extension.generated_material_ui.RNA_PROPERTIES,
    *extension.single_object_operator.RNA_PROPERTIES,
)


def _operator_class(bl_idname: str):
    namespace, operator_name = bl_idname.split(".", 1)
    return bpy.types.Operator.bl_rna_get_subclass_py(
        f"{namespace.upper()}_OT_{operator_name}"
    )


def _register_steps() -> tuple[tuple[str, object, object], ...]:
    completed: list[tuple[str, object, object]] = []
    try:
        for step in extension.REGISTRATION_STEPS:
            _label, register_callback, _unregister_callback = step
            register_callback()
            completed.append(step)
    except Exception:
        for _label, _register_callback, unregister_callback in reversed(completed):
            unregister_callback()
        raise
    return tuple(completed)


def _unregister_steps(completed: tuple[tuple[str, object, object], ...]) -> None:
    failures: list[str] = []
    for label, _register_callback, unregister_callback in reversed(completed):
        try:
            unregister_callback()
        except Exception as exc:
            failures.append(f"{label}: {type(exc).__name__}: {exc}")
    if failures:
        raise AssertionError("registration cleanup failed: " + "; ".join(failures))


def _assert_registered() -> None:
    for registration in RNA_REGISTRATIONS:
        assert hasattr(registration.owner, registration.name), registration.name
    handler = extension.ui.a1_readiness_depsgraph_update_post
    assert tuple(bpy.app.handlers.depsgraph_update_post).count(handler) == 1
    assert _operator_class("object.spine2d_single_export") is not None
    assert _operator_class("object.spine2d_multi_export") is not None
    assert _operator_class("object.save_uv_as_json") is not None


def _assert_unregistered() -> None:
    for registration in RNA_REGISTRATIONS:
        assert not hasattr(registration.owner, registration.name), registration.name
    handler = extension.ui.a1_readiness_depsgraph_update_post
    assert handler not in bpy.app.handlers.depsgraph_update_post
    assert _operator_class("object.spine2d_single_export") is None
    assert _operator_class("object.spine2d_multi_export") is None
    assert _operator_class("object.save_uv_as_json") is None


def test_registration_survives_twenty_cycles_without_handler_or_rna_growth(
    clean_blender_data,
):
    _assert_unregistered()
    baseline_handlers = tuple(bpy.app.handlers.depsgraph_update_post)

    for _cycle in range(20):
        completed = _register_steps()
        try:
            _assert_registered()
        finally:
            _unregister_steps(completed)
        _assert_unregistered()
        assert tuple(bpy.app.handlers.depsgraph_update_post) == baseline_handlers


def test_rewrite_modules_import_and_reload_without_side_effects_in_fresh_process(
    clean_blender_data,
):
    root = Path(__file__).resolve().parents[1]
    script = r'''
import importlib
import pkgutil
import sys

import bpy

PACKAGE = "Blender_to_Spine2D_Mesh_Exporter"
ALLOWED_ROOTS = {"application", "blender_adapter", "domain", "infrastructure"}
ALLOWED_TOP_LEVEL = {
    "addon_preferences",
    "config",
    "repolish_ui",
    "single_object_operator",
    "ui",
}


def id_signature():
    result = []
    for name in (
        "objects", "meshes", "collections", "materials", "images",
        "node_groups", "actions", "cameras", "lights", "curves", "armatures",
    ):
        collection = getattr(bpy.data, name, None)
        if collection is not None:
            result.append((name, tuple(sorted(item.name_full for item in collection))))
    return tuple(result)


def registration_absent(extension):
    handler = extension.ui.a1_readiness_depsgraph_update_post
    assert handler not in bpy.app.handlers.depsgraph_update_post
    assert not hasattr(bpy.types.Object, "spine2d_bake_settings")
    assert not hasattr(bpy.types.Object, "spine2d_connect_settings")
    for registration in (
        *extension.CONFIG_RNA_PROPERTIES,
        *extension.ui.RNA_PROPERTIES,
        *extension.generated_material_ui.RNA_PROPERTIES,
        *extension.single_object_operator.RNA_PROPERTIES,
    ):
        assert not hasattr(registration.owner, registration.name), registration.name


def register_steps(extension):
    completed = []
    try:
        for step in extension.REGISTRATION_STEPS:
            step[1]()
            completed.append(step)
        return completed
    except Exception:
        for step in reversed(completed):
            step[2]()
        raise


def unregister_steps(completed):
    failures = []
    for label, _register, unregister in reversed(completed):
        try:
            unregister()
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    assert not failures, failures


before = id_signature()
baseline_handlers = tuple(bpy.app.handlers.depsgraph_update_post)
extension = importlib.import_module(PACKAGE)
registration_absent(extension)

for info in pkgutil.walk_packages(extension.__path__, extension.__name__ + "."):
    relative = info.name[len(extension.__name__) + 1:]
    root_name = relative.split(".", 1)[0]
    if root_name in ALLOWED_ROOTS or relative in ALLOWED_TOP_LEVEL:
        importlib.import_module(info.name)

assert id_signature() == before
assert tuple(bpy.app.handlers.depsgraph_update_post) == baseline_handlers
registration_absent(extension)

owner_names = tuple(module.__name__ for module in extension.MODULES)
for order in (owner_names, tuple(reversed(owner_names))):
    for name in order:
        importlib.reload(sys.modules[name])
    extension = importlib.reload(sys.modules[PACKAGE])
    registration_absent(extension)
    completed = register_steps(extension)
    try:
        handler = extension.ui.a1_readiness_depsgraph_update_post
        assert tuple(bpy.app.handlers.depsgraph_update_post).count(handler) == 1
    finally:
        unregister_steps(completed)
    registration_absent(extension)
    assert id_signature() == before
    assert tuple(bpy.app.handlers.depsgraph_update_post) == baseline_handlers
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(root), environment.get("PYTHONPATH", ""))
        if value
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"isolated import/reload process failed with {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
