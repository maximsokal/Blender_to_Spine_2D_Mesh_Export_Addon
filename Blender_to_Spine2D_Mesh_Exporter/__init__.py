# pylint: disable=import-error
"""Main entry point for the Blender 5.2+ Rewrite extension."""

from __future__ import annotations

import logging
from typing import Any, Callable, Tuple


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
    from . import addon_preferences, config, repolish_ui, single_object_operator, ui
    from .blender_adapter import generated_material_ui, scene_properties
    from .infrastructure.blender_registration import (
        RegistrationCleanupAction,
        RnaPropertyRegistration,
        register_rna_properties_transactionally,
        rna_property_cleanup_actions,
        unregister_all_best_effort,
    )
    from .infrastructure.blender_version import require_supported_blender_runtime

    AddonLoggingSettings = addon_preferences.AddonLoggingSettings
    CLASSES_TO_REGISTER = addon_preferences.CLASSES_TO_REGISTER
    LoggingModuleSettings = addon_preferences.LoggingModuleSettings
    ModelToSpine2DAddonPreferences = addon_preferences.ModelToSpine2DAddonPreferences
    SPINE2D_OT_RefreshLoggingModules = (
        addon_preferences.SPINE2D_OT_RefreshLoggingModules
    )
    initialize_logging_preferences = addon_preferences.initialize_logging_preferences

    # Only owners of the Blender 5.2+ Rewrite runtime are imported at startup.
    # Legacy implementation modules are deliberately outside this extension surface.
    MODULES = (
        addon_preferences,
        ui,
        repolish_ui,
        generated_material_ui,
        single_object_operator,
    )

    CONFIG_RNA_PROPERTIES = tuple(
        RnaPropertyRegistration(
            owner=bpy.types.Scene,
            name=name,
            value=prop,
        )
        for name, prop in scene_properties.PROPERTIES
    )

    RegistrationCallback = Callable[[], None]
    RegistrationStep = Tuple[str, RegistrationCallback, RegistrationCallback]

    def _module_owns_runtime_registration(module: Any) -> bool:
        return module in MODULES

    def _register_config_rna() -> None:
        register_rna_properties_transactionally(CONFIG_RNA_PROPERTIES)

    def _unregister_config_rna() -> None:
        unregister_all_best_effort(
            rna_property_cleanup_actions(CONFIG_RNA_PROPERTIES),
            operation="config RNA unregistration",
        )

    REGISTRATION_STEPS: tuple[RegistrationStep, ...] = (
        (
            "addon preferences",
            addon_preferences.register,
            addon_preferences.unregister,
        ),
        (
            "Scene RNA properties",
            _register_config_rna,
            _unregister_config_rna,
        ),
        (
            "UI",
            ui.register,
            ui.unregister,
        ),
        (
            "Re-Polish UI",
            repolish_ui.register,
            repolish_ui.unregister,
        ),
        (
            "generated material UI",
            generated_material_ui.register,
            generated_material_ui.unregister,
        ),
        (
            "single-object operator",
            single_object_operator.register,
            single_object_operator.unregister,
        ),
    )

    def _registration_cleanup_actions(
        completed_steps: tuple[RegistrationStep, ...],
    ) -> tuple[RegistrationCleanupAction, ...]:
        return tuple(
            RegistrationCleanupAction(
                label=label,
                callback=unregister_callback,
            )
            for label, _register_callback, unregister_callback in reversed(
                completed_steps
            )
        )

    def register() -> None:
        """Register the complete Blender 5.2+ Rewrite extension transactionally."""

        require_supported_blender_runtime(bpy)
        config._setup_default_logging()
        logger.debug("Registering Blender_to_Spine2D_Mesh_Exporter Rewrite")

        completed: list[RegistrationStep] = []
        try:
            for step in REGISTRATION_STEPS:
                _label, register_callback, _unregister_callback = step
                register_callback()
                completed.append(step)

            prefs = bpy.context.preferences.addons[__name__].preferences
            initialize_logging_preferences(prefs)
            config.setup_logging()
            logger.info("User logging and diagnostics preferences applied")
        except Exception as exc:
            logger.exception("Rewrite extension registration failed")
            unregister_all_best_effort(
                _registration_cleanup_actions(tuple(completed)),
                operation="Rewrite extension registration rollback",
                primary_error=exc,
            )
            raise

    def unregister() -> None:
        """Run every Rewrite owner cleanup in reverse order before reporting failures."""

        logger.debug("Unregistering Blender_to_Spine2D_Mesh_Exporter Rewrite")
        unregister_all_best_effort(
            _registration_cleanup_actions(REGISTRATION_STEPS),
            operation="Rewrite extension unregistration",
        )

else:
    MODULES: tuple[Any, ...] = ()
    CLASSES_TO_REGISTER: tuple[Any, ...] = ()
    CONFIG_RNA_PROPERTIES: tuple[Any, ...] = ()
    REGISTRATION_STEPS: tuple[Any, ...] = ()

    def _module_owns_runtime_registration(_module: Any) -> bool:
        return False

    def initialize_logging_preferences(_prefs: Any) -> tuple[str, ...]:
        return ()

    def register() -> None:
        raise RuntimeError("Blender bpy module is required to register the extension")

    def unregister() -> None:
        return None


if __name__ == "__main__" and bpy is not None:
    register()
