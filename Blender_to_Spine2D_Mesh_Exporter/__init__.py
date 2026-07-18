# pylint: disable=import-error
"""Main entry point for the Blender to Spine2D Mesh Exporter add-on."""

from __future__ import annotations

import logging
from typing import Any


bl_info = {
    "name": "Blender to Spine2D Mesh Exporter",
    "author": "Maxim Sokolenko",
    "version": (0, 23, 0),
    "blender": (4, 4, 0),
    "location": "View3D > UI > Blender to Spine2D Mesh Exporter",
    "description": "Converts 3D objects into a Spine2D JSON structure",
    "warning": "",
    "category": "3D View",
}


logger = logging.getLogger(__package__ or __name__)

try:
    import bpy as _bpy  # type: ignore
except ModuleNotFoundError:
    _bpy = None

if _bpy is not None and all(
    hasattr(_bpy, attribute) for attribute in ("types", "props", "utils", "context")
):
    bpy = _bpy
else:
    bpy = None


if bpy is not None:
    from . import config
    from .legacy_loader import install_legacy_multi_facade

    install_legacy_multi_facade()
    from . import single_object_operator, ui
    from .addon_preferences import (
        AddonLoggingSettings,
        CLASSES_TO_REGISTER,
        LoggingModuleSettings,
        ModelToSpine2DAddonPreferences,
        SPINE2D_OT_RefreshLoggingModules,
        WM_OT_UninstallAddon,
        initialize_logging_preferences,
    )

    # Only modules that own live Blender classes/properties are imported during add-on startup.
    # Legacy implementation modules are loaded by ``legacy_loader`` only after explicit Legacy use.
    MODULES = (ui, single_object_operator)

    def _module_owns_runtime_registration(module: Any) -> bool:
        return module in MODULES

    def register() -> None:
        config._setup_default_logging()
        logger.debug("Registering Blender_to_Spine2D_Mesh_Exporter")

        registered_classes: list[type] = []
        registered_modules: list[Any] = []
        try:
            for cls in CLASSES_TO_REGISTER:
                bpy.utils.register_class(cls)
                registered_classes.append(cls)

            for module in MODULES:
                if not _module_owns_runtime_registration(module):
                    continue
                register_module = getattr(module, "register", None)
                if callable(register_module):
                    register_module()
                    registered_modules.append(module)

            prefs = bpy.context.preferences.addons[__name__].preferences
            initialize_logging_preferences(prefs)
            config.setup_logging()
            logger.info("User logging and diagnostics preferences applied")
        except Exception:
            logger.exception("Addon registration failed")
            for module in reversed(registered_modules):
                try:
                    unregister_module = getattr(module, "unregister", None)
                    if callable(unregister_module):
                        unregister_module()
                except Exception:
                    logger.exception(
                        "Registration rollback failed for module %s",
                        module.__name__,
                    )
            for cls in reversed(registered_classes):
                try:
                    bpy.utils.unregister_class(cls)
                except Exception:
                    logger.exception("Registration rollback failed for %s", cls.__name__)
            raise

    def unregister() -> None:
        logger.debug("Unregistering Blender_to_Spine2D_Mesh_Exporter")
        errors: list[Exception] = []
        for module in reversed(MODULES):
            if not _module_owns_runtime_registration(module):
                continue
            try:
                unregister_module = getattr(module, "unregister", None)
                if callable(unregister_module):
                    unregister_module()
            except Exception as exc:
                errors.append(exc)
                logger.exception("Failed to unregister module %s", module.__name__)
        for cls in reversed(CLASSES_TO_REGISTER):
            try:
                bpy.utils.unregister_class(cls)
            except Exception as exc:
                errors.append(exc)
                logger.exception("Failed to unregister class %s", cls.__name__)
        if errors:
            raise RuntimeError(
                f"Addon unregistration failed {len(errors)} time(s)"
            ) from errors[0]

else:
    MODULES: tuple[Any, ...] = ()
    CLASSES_TO_REGISTER: tuple[Any, ...] = ()

    def _module_owns_runtime_registration(_module: Any) -> bool:
        return False

    def initialize_logging_preferences(_prefs: Any) -> tuple[str, ...]:
        return ()

    def register() -> None:
        raise RuntimeError("Blender bpy module is required to register the add-on")

    def unregister() -> None:
        return None


if __name__ == "__main__" and bpy is not None:
    register()
