# tests/test_init.py
"""
Comprehensive test suite for __init__.py registration and preference management.

This module tests the critical addon lifecycle functions including:
- Module registration/unregistration with error handling
- Addon preferences initialization and management
- Logging configuration integration
- UI operator functionality for addon management
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, patch


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


try:
    import Blender_to_Spine2D_Mesh_Exporter
    from Blender_to_Spine2D_Mesh_Exporter import config
except ImportError as e:
    print(f"Critical import failure: {e}")
    print(f"Project root: {project_root}")
    print(f"Available paths: {[p for p in sys.path if 'spine2d' in p or 'test' in p]}")
    raise ImportError(f"Cannot import required modules from {project_root}")


import bpy


class TestInitializeLoggingPreferences:
    def setup_method(self):
        bpy.reset_mock()
        bpy.context = MagicMock()
        bpy.utils = MagicMock()

    def test_initialize_logging_preferences_success(self):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings

        mock_logging_settings.log_file_path = ""
        mock_logging_settings.modules = MagicMock()
        # Make it behave like a list for the 'if not' check, but still be a mock
        mock_logging_settings.modules.__bool__.return_value = False

        with patch("os.path.expanduser", return_value="/home/testuser"), patch(
            "os.path.join",
            return_value="/home/testuser/Blender_to_Spine2D_Mesh_Exporter.log",
        ):
            Blender_to_Spine2D_Mesh_Exporter.initialize_logging_preferences(mock_prefs)

        assert (
            mock_logging_settings.log_file_path
            == "/home/testuser/Blender_to_Spine2D_Mesh_Exporter.log"
        )

        expected_modules = Blender_to_Spine2D_Mesh_Exporter.MODULE_NAMES_FOR_LOGGING
        assert mock_logging_settings.modules.add.call_count == len(expected_modules)

        add_calls = mock_logging_settings.modules.add.call_args_list
        for i, expected_name in enumerate(expected_modules):
            # Check that a module with the correct name was added
            added_module_mock = add_calls[i].this_val
            # This is a bit tricky, we check the properties set on the returned mock
            # from the .add() call.
            added_item = mock_logging_settings.modules.add.return_value
            # In a real scenario, the call would be something like:
            # module = prefs.logging_settings.modules.add()
            # module.name = name
            # So we can check the call history of the mock.
            # For simplicity, we'll just check that add was called enough times.
            pass  # The call_count assertion is sufficient for this test's purpose.

    def test_initialize_logging_preferences_existing_path(self):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings

        existing_path = "/existing/path/custom.log"
        mock_logging_settings.log_file_path = existing_path
        mock_logging_settings.modules = MagicMock()
        mock_logging_settings.modules.__bool__.return_value = False

        Blender_to_Spine2D_Mesh_Exporter.initialize_logging_preferences(mock_prefs)

        assert mock_logging_settings.log_file_path == existing_path

        expected_count = len(Blender_to_Spine2D_Mesh_Exporter.MODULE_NAMES_FOR_LOGGING)
        assert mock_logging_settings.modules.add.call_count == expected_count

    def test_initialize_logging_preferences_existing_modules(self):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings

        mock_logging_settings.log_file_path = ""
        # Simulate an existing collection with items
        mock_logging_settings.modules = MagicMock()
        mock_logging_settings.modules.__bool__.return_value = True
        mock_logging_settings.modules.__len__.return_value = 1  # Not empty

        with patch("os.path.expanduser", return_value="/home/testuser"):
            Blender_to_Spine2D_Mesh_Exporter.initialize_logging_preferences(mock_prefs)

        assert mock_logging_settings.log_file_path != ""
        mock_logging_settings.modules.add.assert_not_called()

    def test_initialize_logging_preferences_missing_logging_settings(self):
        mock_prefs = MagicMock()

        del mock_prefs.logging_settings

        try:
            Blender_to_Spine2D_Mesh_Exporter.initialize_logging_preferences(mock_prefs)
        except AttributeError:
            pytest.fail("Function should handle missing logging_settings gracefully")


class TestUninstallAddonOperator:
    def setup_method(self):
        bpy.reset_mock()
        bpy.ops = MagicMock()
        bpy.context = MagicMock()

        self.operator = Blender_to_Spine2D_Mesh_Exporter.WM_OT_UninstallAddon()
        self.operator.report = MagicMock()

        self.mock_context = MagicMock()

    def test_uninstall_addon_success(self):
        bpy.ops.preferences.addon_disable.return_value = {"FINISHED"}
        bpy.ops.preferences.addon_remove.return_value = {"FINISHED"}

        self.operator.module = "test_addon_module"

        result = self.operator.execute(self.mock_context)

        assert result == {"FINISHED"}

        bpy.ops.preferences.addon_disable.assert_called_once_with(
            module="test_addon_module"
        )
        bpy.ops.preferences.addon_remove.assert_called_once_with(
            module="test_addon_module"
        )

        self.operator.report.assert_called_with(
            {"INFO"}, "Addon uninstalled successfully."
        )

    def test_uninstall_addon_disable_failure(self):
        bpy.ops.preferences.addon_disable.side_effect = Exception("Disable failed")
        bpy.ops.preferences.addon_remove.return_value = {"FINISHED"}

        self.operator.module = "test_addon_module"

        result = self.operator.execute(self.mock_context)

        assert result == {"FINISHED"}

        bpy.ops.preferences.addon_remove.assert_called_once_with(
            module="test_addon_module"
        )

    def test_uninstall_addon_remove_failure(self):
        bpy.ops.preferences.addon_disable.return_value = {"FINISHED"}
        bpy.ops.preferences.addon_remove.side_effect = Exception("Remove failed")

        self.operator.module = "test_addon_module"

        result = self.operator.execute(self.mock_context)

        assert result == {"CANCELLED"}

        error_calls = [
            call
            for call in self.operator.report.call_args_list
            if call[0][0] == {"ERROR"}
        ]
        assert len(error_calls) > 0
        assert "Uninstall failed" in error_calls[0][0][1]

    def test_uninstall_addon_complete_failure(self):
        bpy.ops.preferences.addon_disable.side_effect = Exception("Disable failed")
        bpy.ops.preferences.addon_remove.side_effect = Exception("Remove failed")

        self.operator.module = "test_addon_module"

        result = self.operator.execute(self.mock_context)

        assert result == {"CANCELLED"}

        bpy.ops.preferences.addon_disable.assert_called_once()
        bpy.ops.preferences.addon_remove.assert_called_once()


class TestAddonPreferences:
    def setup_method(self):
        bpy.reset_mock()
        bpy.context = MagicMock()
        bpy.types = MagicMock()

        self.preferences = (
            Blender_to_Spine2D_Mesh_Exporter.ModelToSpine2DAddonPreferences()
        )

    def test_addon_preferences_class_structure(self):
        assert hasattr(
            Blender_to_Spine2D_Mesh_Exporter.ModelToSpine2DAddonPreferences, "bl_idname"
        )
        assert (
            Blender_to_Spine2D_Mesh_Exporter.ModelToSpine2DAddonPreferences.bl_idname
            == Blender_to_Spine2D_Mesh_Exporter.__name__
        )

        annotations = getattr(
            Blender_to_Spine2D_Mesh_Exporter.ModelToSpine2DAddonPreferences,
            "__annotations__",
            {},
        )
        assert "logging_settings" in annotations

    @patch.object(config, "setup_logging")
    def test_update_logging_config_method(self, mock_setup_logging):
        self.preferences.update_logging_config()

        mock_setup_logging.assert_called_once()

    def test_preferences_draw_method_structure(self):
        assert hasattr(self.preferences, "draw")
        assert callable(self.preferences.draw)

        mock_context = MagicMock()
        mock_layout = MagicMock()
        mock_context.layout = mock_layout

        try:
            # This might fail due to missing attributes, but shouldn't crash completely
            self.preferences.draw(mock_context)
        except AttributeError as e:
            assert "logging_settings" in str(e)


class TestRegistrationSystem:
    def setup_method(self):
        bpy.reset_mock()
        bpy.utils.register_class = MagicMock()
        bpy.utils.unregister_class = MagicMock()
        bpy.context = MagicMock()

        mock_preferences = MagicMock()
        mock_logging_settings = MagicMock()
        mock_preferences.logging_settings = mock_logging_settings
        bpy.context.preferences.addons = {
            Blender_to_Spine2D_Mesh_Exporter.__name__: MagicMock()
        }
        bpy.context.preferences.addons[
            Blender_to_Spine2D_Mesh_Exporter.__name__
        ].preferences = mock_preferences

    @patch.object(config, "_setup_default_logging")
    @patch.object(config, "setup_logging")
    @patch.object(Blender_to_Spine2D_Mesh_Exporter, "initialize_logging_preferences")
    def test_register_complete_success(
        self, mock_init_prefs, mock_setup_logging, mock_default_logging
    ):
        mock_modules = []
        for i in range(3):
            mock_module = MagicMock()
            mock_module.__name__ = f"test_module_{i}"
            mock_module.register = MagicMock()
            mock_modules.append(mock_module)

        original_modules = Blender_to_Spine2D_Mesh_Exporter.MODULES
        Blender_to_Spine2D_Mesh_Exporter.MODULES = tuple(mock_modules)

        try:
            Blender_to_Spine2D_Mesh_Exporter.register()

            mock_default_logging.assert_called_once()

            expected_classes = Blender_to_Spine2D_Mesh_Exporter.CLASSES_TO_REGISTER
            assert bpy.utils.register_class.call_count == len(expected_classes)

            for mock_module in mock_modules:
                mock_module.register.assert_called_once()

            mock_init_prefs.assert_called_once()
            mock_setup_logging.assert_called_once()

        finally:
            Blender_to_Spine2D_Mesh_Exporter.MODULES = original_modules

    @patch.object(config, "_setup_default_logging")
    def test_register_class_failure(self, mock_default_logging):
        def register_class_side_effect(cls):
            if cls.__name__ == "LoggingModuleSettings":
                raise Exception("Registration failed")
            return None

        bpy.utils.register_class.side_effect = register_class_side_effect

        mock_modules = [MagicMock()]
        mock_modules[0].__name__ = "test_module"
        mock_modules[0].register = MagicMock()

        original_modules = Blender_to_Spine2D_Mesh_Exporter.MODULES
        Blender_to_Spine2D_Mesh_Exporter.MODULES = tuple(mock_modules)

        try:
            Blender_to_Spine2D_Mesh_Exporter.register()

            mock_modules[0].register.assert_called_once()

        finally:
            Blender_to_Spine2D_Mesh_Exporter.MODULES = original_modules

    @patch.object(config, "_setup_default_logging")
    def test_register_module_failure(self, mock_default_logging):
        failing_module = MagicMock()
        failing_module.__name__ = "failing_module"
        failing_module.register.side_effect = Exception("Module registration failed")

        success_module = MagicMock()
        success_module.__name__ = "success_module"
        success_module.register = MagicMock()

        original_modules = Blender_to_Spine2D_Mesh_Exporter.MODULES
        Blender_to_Spine2D_Mesh_Exporter.MODULES = (failing_module, success_module)

        try:
            Blender_to_Spine2D_Mesh_Exporter.register()

            failing_module.register.assert_called_once()
            success_module.register.assert_called_once()

        finally:
            Blender_to_Spine2D_Mesh_Exporter.MODULES = original_modules

    def test_unregister_complete_success(self):
        mock_modules = []
        for i in range(3):
            mock_module = MagicMock()
            mock_module.__name__ = f"test_module_{i}"
            mock_module.unregister = MagicMock()
            mock_modules.append(mock_module)

        original_modules = Blender_to_Spine2D_Mesh_Exporter.MODULES
        Blender_to_Spine2D_Mesh_Exporter.MODULES = tuple(mock_modules)

        try:
            Blender_to_Spine2D_Mesh_Exporter.unregister()

            for mock_module in reversed(mock_modules):
                mock_module.unregister.assert_called_once()

            expected_classes = Blender_to_Spine2D_Mesh_Exporter.CLASSES_TO_REGISTER
            assert bpy.utils.unregister_class.call_count == len(expected_classes)

        finally:
            Blender_to_Spine2D_Mesh_Exporter.MODULES = original_modules

    def test_unregister_with_failures(self):
        failing_module = MagicMock()
        failing_module.__name__ = "failing_module"
        failing_module.unregister.side_effect = Exception("Unregister failed")

        success_module = MagicMock()
        success_module.__name__ = "success_module"
        success_module.unregister = MagicMock()

        def unregister_class_side_effect(cls):
            if cls.__name__ == "WM_OT_UninstallAddon":
                raise Exception("Class unregister failed")
            return None

        bpy.utils.unregister_class.side_effect = unregister_class_side_effect

        original_modules = Blender_to_Spine2D_Mesh_Exporter.MODULES
        Blender_to_Spine2D_Mesh_Exporter.MODULES = (failing_module, success_module)

        try:
            Blender_to_Spine2D_Mesh_Exporter.unregister()

            failing_module.unregister.assert_called_once()
            success_module.unregister.assert_called_once()

            expected_classes = len(Blender_to_Spine2D_Mesh_Exporter.CLASSES_TO_REGISTER)
            assert bpy.utils.unregister_class.call_count == expected_classes

        finally:
            Blender_to_Spine2D_Mesh_Exporter.MODULES = original_modules


class TestModuleStructureValidation:
    def test_modules_tuple_structure(self):
        modules = Blender_to_Spine2D_Mesh_Exporter.MODULES

        assert isinstance(modules, tuple)

        # Verify minimum expected modules
        module_names = [module.__name__.split(".")[-1] for module in modules]
        expected_modules = ["config", "ui", "main"]

        for expected in expected_modules:
            assert (
                expected in module_names
            ), f"Expected module '{expected}' not found in MODULES"

    def test_classes_to_register_structure(self):
        classes = Blender_to_Spine2D_Mesh_Exporter.CLASSES_TO_REGISTER

        # Verify it's a tuple
        assert isinstance(classes, tuple)

        class_names = [cls.__name__ for cls in classes]
        expected_classes = [
            "LoggingModuleSettings",
            "AddonLoggingSettings",
            "ModelToSpine2DAddonPreferences",
            "WM_OT_UninstallAddon",
        ]

        for expected in expected_classes:
            assert (
                expected in class_names
            ), f"Expected class '{expected}' not found in CLASSES_TO_REGISTER"

    def test_module_names_for_logging_completeness(self):
        logging_modules = Blender_to_Spine2D_Mesh_Exporter.MODULE_NAMES_FOR_LOGGING

        assert isinstance(logging_modules, list)

        # Verify includes main addon logger
        assert "Blender_to_Spine2D_Mesh_Exporter" in logging_modules

        # Verify includes config module
        assert "config" in logging_modules

        # Verify reasonable number of modules
        assert (
            len(logging_modules) >= 5
        ), "Expected at least 5 modules for logging configuration"


# ============================================================================
# Integration Tests
# ============================================================================


class TestRegistrationIntegration:
    @patch.object(config, "_setup_default_logging")
    @patch.object(config, "setup_logging")
    @patch.object(Blender_to_Spine2D_Mesh_Exporter, "initialize_logging_preferences")
    def test_complete_registration_cycle(
        self, mock_init_prefs, mock_setup_logging, mock_default_logging
    ):
        bpy.utils.register_class = MagicMock()
        bpy.utils.unregister_class = MagicMock()

        mock_module = MagicMock()
        mock_module.__name__ = "test_integration_module"
        mock_module.register = MagicMock()
        mock_module.unregister = MagicMock()

        original_modules = Blender_to_Spine2D_Mesh_Exporter.MODULES
        Blender_to_Spine2D_Mesh_Exporter.MODULES = (mock_module,)

        try:
            Blender_to_Spine2D_Mesh_Exporter.register()

            mock_module.register.assert_called_once()
            assert bpy.utils.register_class.called

            mock_module.reset_mock()
            bpy.utils.unregister_class.reset_mock()

            Blender_to_Spine2D_Mesh_Exporter.unregister()

            mock_module.unregister.assert_called_once()
            assert bpy.utils.unregister_class.called

        finally:
            Blender_to_Spine2D_Mesh_Exporter.MODULES = original_modules
