# tests/test_config.py
import os
import sys
import logging
from unittest.mock import MagicMock, patch
import importlib
import pytest


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


try:
    from Blender_to_Spine2D_Mesh_Exporter import config
except ImportError as e:
    print(f"Failed to import Blender_to_Spine2D_Mesh_Exporter.config: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path[:3]}")
    raise


import bpy


def setup_function():
    bpy.context.scene = MagicMock()
    bpy.data = MagicMock()

    bpy.data.filepath = ""
    bpy.data.is_saved = False
    importlib.reload(config)


@patch("os.path.dirname")
@patch("os.path.expanduser")
def test_get_default_output_dir(mock_expanduser, mock_dirname):
    bpy.data.is_saved = True
    bpy.data.filepath = "/path/to/my/project.blend"
    mock_dirname.return_value = "/path/to/my"
    assert config.get_default_output_dir() == "/path/to/my"
    mock_dirname.assert_called_with("/path/to/my/project.blend")

    bpy.data.is_saved = False
    bpy.data.filepath = ""
    mock_expanduser.return_value = "/home/user"
    assert config.get_default_output_dir() == "/home/user"

    bpy.data.is_saved = True
    bpy.data.filepath = None
    assert config.get_default_output_dir() == "/home/user"


def test_set_frames_for_render():
    mock_self = {}

    type(bpy.context.scene).frame_end = property(lambda self: 100)

    config.set_frames_for_render(mock_self, 50)
    assert mock_self["spine2d_frames_for_render"] == 50

    config.set_frames_for_render(mock_self, 150)
    assert mock_self["spine2d_frames_for_render"] == 100

    config.set_frames_for_render(mock_self, -10)
    assert mock_self["spine2d_frames_for_render"] == 0


class TestShortNameFormatter:
    def setup_method(self):
        self.formatter = config.ShortNameFormatter()

    def test_format_simple_logger_name(self):
        record = MagicMock()
        record.name = "Blender_to_Spine2D_Mesh_Exporter.config"
        record.getMessage.return_value = "Test message"
        record.levelname = "INFO"
        record.created = 1234567890.123

        with patch.object(logging.Formatter, "format", return_value="formatted_output"):
            result = self.formatter.format(record)

        assert record.name == "config"
        assert result == "formatted_output"

    def test_format_complex_logger_name(self):
        record = MagicMock()
        record.name = (
            "bl_ext.user_default.Model_to_Spine2D_Mesh.texture_baker_integration"
        )
        record.getMessage.return_value = "Complex test message"

        with patch.object(
            logging.Formatter, "format", return_value="complex_formatted"
        ):
            result = self.formatter.format(record)

        assert record.name == "texture_baker_integration"
        assert result == "complex_formatted"

    def test_format_single_component_name(self):
        record = MagicMock()
        record.name = "Blender_to_Spine2D_Mesh_Exporter"

        with patch.object(logging.Formatter, "format", return_value="single_component"):
            result = self.formatter.format(record)

        assert record.name == "Blender_to_Spine2D_Mesh_Exporter"
        assert result == "single_component"


class TestUniformScaleCalculation:
    def test_calc_uniform_scale_default_mode(self):
        result = config.calc_uniform_scale(1024, 512)
        expected = (1024 + 512) / 2
        assert result == expected

        result = config.calc_uniform_scale(1024, 1024)
        assert result == 1024.0

        result = config.calc_uniform_scale(2048, 256)
        expected = (2048 + 256) / 2
        assert result == expected

    def test_calc_uniform_scale_different_modes(self):
        width, height = 1024, 512

        result = config.calc_uniform_scale(width, height, mode="average")
        assert result == (width + height) / 2

        result_default = config.calc_uniform_scale(width, height)
        result_explicit = config.calc_uniform_scale(width, height, mode="average")
        assert result_default == result_explicit

    def test_calc_uniform_scale_edge_cases(self):
        result = config.calc_uniform_scale(0, 100)
        assert result == 50.0

        result = config.calc_uniform_scale(0, 0)
        assert result == 0.0

        result = config.calc_uniform_scale(8192, 8192)
        assert result == 8192.0


class TestUpdateLoggingConfig:
    @patch("bpy.context.preferences.addons")
    def test_update_logging_config_success(self, mock_addons):
        mock_preferences = MagicMock()
        mock_preferences.update_logging_config = MagicMock()

        addon_prefs = MagicMock()
        addon_prefs.preferences = mock_preferences
        mock_addons.__getitem__.return_value = addon_prefs
        mock_addons.__contains__.return_value = True
        config.__package__ = "Blender_to_Spine2D_Mesh_Exporter"

        mock_context = MagicMock()
        mock_context.preferences.addons = mock_addons

        # Pass a mock object for 'self' that is not the main logger
        mock_self = MagicMock()
        mock_self.name = "some_other_module"
        config._update_logging_config(mock_self, mock_context)

        mock_preferences.update_logging_config.assert_called_once()

    @patch("bpy.context.preferences.addons")
    def test_update_logging_config_missing_addon(self, mock_addons):
        mock_addons.__contains__.return_value = False

        mock_context = MagicMock()
        mock_context.preferences.addons = mock_addons

        try:
            config._update_logging_config(None, mock_context)
        except Exception as e:
            pytest.fail(f"Update callback should handle missing addon gracefully: {e}")

    @patch("bpy.context.preferences.addons")
    def test_update_logging_config_global_change(self, mock_addons):
        """
        Tests that changing the main 'Blender_to_Spine2D_Mesh_Exporter' logger updates all other loggers.
        """
        # 1. Setup Mocks
        mock_preferences = MagicMock()
        mock_preferences.update_logging_config = MagicMock()
        log_settings = MagicMock()
        mock_preferences.logging_settings = log_settings

        # Create mock module settings with real attributes
        mock_main_logger = MagicMock()
        mock_main_logger.name = "Blender_to_Spine2D_Mesh_Exporter"
        mock_main_logger.level = "ERROR"

        mock_config_logger = MagicMock()
        mock_config_logger.name = "config"
        mock_config_logger.level = "ERROR"

        mock_ui_logger = MagicMock()
        mock_ui_logger.name = "ui"
        mock_ui_logger.level = "ERROR"

        log_settings.modules = [mock_main_logger, mock_config_logger, mock_ui_logger]

        # Setup context to find these preferences
        addon_prefs = MagicMock()
        addon_prefs.preferences = mock_preferences
        mock_addons.__getitem__.return_value = addon_prefs
        mock_addons.__contains__.return_value = True
        config.__package__ = "Blender_to_Spine2D_Mesh_Exporter"
        mock_context = MagicMock()
        mock_context.preferences.addons = mock_addons

        # 2. Action: Simulate changing the main logger's level to DEBUG
        mock_main_logger.level = "DEBUG"
        config._update_logging_config(self=mock_main_logger, context=mock_context)

        # 3. Assertions
        # All loggers should now be DEBUG
        assert mock_main_logger.level == "DEBUG"
        assert mock_config_logger.level == "DEBUG"
        assert mock_ui_logger.level == "DEBUG"
        # The final update function must be called
        mock_preferences.update_logging_config.assert_called_once()

    @patch("bpy.context.preferences.addons")
    def test_update_logging_config_individual_change(self, mock_addons):
        """
        Tests that changing an individual logger does not affect others.
        """
        # 1. Setup Mocks (with corrected attribute assignment)
        mock_preferences = MagicMock()
        mock_preferences.update_logging_config = MagicMock()
        log_settings = MagicMock()
        mock_preferences.logging_settings = log_settings

        mock_main_logger = MagicMock()
        mock_main_logger.name = "Blender_to_Spine2D_Mesh_Exporter"
        mock_main_logger.level = "ERROR"

        mock_config_logger = MagicMock()
        mock_config_logger.name = "config"
        mock_config_logger.level = "ERROR"

        mock_ui_logger = MagicMock()
        mock_ui_logger.name = "ui"
        mock_ui_logger.level = "ERROR"

        log_settings.modules = [mock_main_logger, mock_config_logger, mock_ui_logger]

        addon_prefs = MagicMock()
        addon_prefs.preferences = mock_preferences
        mock_addons.__getitem__.return_value = addon_prefs
        mock_addons.__contains__.return_value = True
        config.__package__ = "Blender_to_Spine2D_Mesh_Exporter"
        mock_context = MagicMock()
        mock_context.preferences.addons = mock_addons

        # 2. Action: Simulate changing the 'config' logger's level to INFO
        mock_config_logger.level = "INFO"
        config._update_logging_config(self=mock_config_logger, context=mock_context)

        # 3. Assertions
        # Only the changed logger should be INFO
        assert mock_config_logger.level == "INFO"
        # Others should remain ERROR
        assert mock_main_logger.level == "ERROR"
        assert mock_ui_logger.level == "ERROR"
        # The final update function must still be called
        mock_preferences.update_logging_config.assert_called_once()
