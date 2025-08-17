# tests/test_logging_integration.py
"""
Comprehensive Integration Test Suite for Addon Logging System

This module provides end-to-end testing for the addon's logging configuration system,
validating the complete workflow from user preferences to actual log output.

## Test Coverage Areas:
1. User Interface to Logging Configuration Integration
2. File-based Logging Functionality with Real I/O Operations
3. Multi-module Logging Level Configuration
4. Real-time Configuration Updates and Callbacks
5. Error Recovery and Fallback Mechanisms
6. Performance Validation for High-volume Logging

## Integration Test Philosophy:
These tests verify that the logging system components work together correctly
in realistic usage scenarios, complementing the unit tests with system-level validation.
"""
# tests/test_logging_integration.py
import os
import sys
import tempfile
import logging
import time
import threading
from unittest.mock import MagicMock, patch
import pytest
import importlib


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)


try:
    import Blender_to_Spine2D_Mesh_Exporter
    from Blender_to_Spine2D_Mesh_Exporter import config
except ImportError as e:
    print(f"Integration test setup failed - module import error: {e}")
    print("Project structure verification:")
    print(f"  - Project root: {project_root}")
    print(
        f"  - Expected addon path: {os.path.join(project_root, 'Blender_to_Spine2D_Mesh_Exporter')}"
    )
    print(
        f"  - Path exists: {os.path.exists(os.path.join(project_root, 'Blender_to_Spine2D_Mesh_Exporter'))}"
    )
    raise

import bpy


def _close_all_loggers():
    """
    Correctly closes and removes all handlers from all loggers.
    """
    manager = logging.root.manager
    for name in list(manager.loggerDict.keys()):
        if isinstance(manager.loggerDict[name], logging.Logger):
            logger = logging.getLogger(name)
            for h in logger.handlers[:]:
                try:
                    h.flush()
                    h.close()
                except Exception:
                    pass
                logger.removeHandler(h)

    # Also clean up the root logger
    root = logging.getLogger()
    for h in root.handlers[:]:
        try:
            h.flush()
            h.close()
        except Exception:
            pass
        root.removeHandler(h)

    # This is a final safeguard
    logging.shutdown()


class TestLoggingConfigurationIntegration:
    def setup_method(self):
        bpy.reset_mock()

        bpy.context = MagicMock()
        bpy.context.preferences = MagicMock()
        bpy.context.preferences.addons = {}

        bpy.path = MagicMock()
        bpy.path.abspath = lambda x: os.path.abspath(x)  # Use real abspath

        importlib.reload(config)

        self.temp_dir = tempfile.mkdtemp(
            prefix="Blender_to_Spine2D_Mesh_Exporter_logging_test_"
        )
        self.temp_log_file = os.path.join(self.temp_dir, "test_addon.log")

    def teardown_method(self):
        import shutil

        _close_all_loggers()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

        # Clean up logging handlers to avoid interference between tests
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        logging.shutdown()

    @patch("logging.config.dictConfig")
    def test_complete_logging_setup_workflow(self, mock_dict_config):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings

        mock_logging_settings.log_file_path = ""
        mock_logging_settings.modules = MagicMock()
        mock_logging_settings.modules.__bool__.return_value = False
        mock_logging_settings.enable_file_logging = False

        with patch("os.path.expanduser", return_value="/home/testuser"), patch(
            "os.path.join",
            return_value="/home/testuser/Blender_to_Spine2D_Mesh_Exporter.log",
        ):
            Blender_to_Spine2D_Mesh_Exporter.initialize_logging_preferences(mock_prefs)

        assert (
            mock_logging_settings.log_file_path
            == "/home/testuser/Blender_to_Spine2D_Mesh_Exporter.log"
        )
        expected_module_count = len(
            Blender_to_Spine2D_Mesh_Exporter.MODULE_NAMES_FOR_LOGGING
        )
        assert mock_logging_settings.modules.add.call_count == expected_module_count

        mock_logging_settings.enable_file_logging = True
        mock_logging_settings.log_file_path = self.temp_log_file

        mock_modules = []
        test_modules = [
            ("Blender_to_Spine2D_Mesh_Exporter", "DEBUG"),
            ("config", "INFO"),
            ("ui", "WARNING"),
            ("main", "ERROR"),
        ]

        for module_name, level in test_modules:
            mock_module = MagicMock()
            mock_module.name = module_name
            mock_module.level = level
            mock_modules.append(mock_module)

        mock_logging_settings.modules = mock_modules

        addon_name = Blender_to_Spine2D_Mesh_Exporter.__name__
        bpy.context.preferences.addons[addon_name] = MagicMock()
        bpy.context.preferences.addons[addon_name].preferences = mock_prefs

        config.setup_logging()

        mock_dict_config.assert_called_once()

        config_dict = mock_dict_config.call_args[0][0]

        assert "file" in config_dict["handlers"]
        assert config_dict["handlers"]["file"]["filename"] == self.temp_log_file

        loggers = config_dict["loggers"]
        assert "Blender_to_Spine2D_Mesh_Exporter" in loggers
        assert loggers["Blender_to_Spine2D_Mesh_Exporter"]["level"] == "DEBUG"

        # Use the hardcoded addon ID from the manifest
        expected_config_logger = (
            "bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.config"
        )
        assert expected_config_logger in loggers
        assert loggers[expected_config_logger]["level"] == "INFO"

    def test_real_file_logging_integration(self):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings

        mock_logging_settings.enable_file_logging = True
        mock_logging_settings.log_file_path = self.temp_log_file

        mock_module = MagicMock()
        mock_module.name = "Blender_to_Spine2D_Mesh_Exporter"
        mock_module.level = "DEBUG"
        mock_logging_settings.modules = [mock_module]

        addon_name = Blender_to_Spine2D_Mesh_Exporter.__name__
        bpy.context.preferences.addons[addon_name] = MagicMock()
        bpy.context.preferences.addons[addon_name].preferences = mock_prefs

        config.setup_logging()

        test_logger = logging.getLogger("Blender_to_Spine2D_Mesh_Exporter")
        test_messages = [
            ("DEBUG", "Debug message for integration test"),
            ("INFO", "Info message for integration test"),
            ("WARNING", "Warning message for integration test"),
            ("ERROR", "Error message for integration test"),
        ]

        for level, message in test_messages:
            getattr(test_logger, level.lower())(message)

        for handler in logging.getLogger().handlers:
            handler.flush()
            handler.close()

        assert os.path.exists(self.temp_log_file), "Log file was not created"

        with open(self.temp_log_file, "r", encoding="utf-8") as log_file:
            log_content = log_file.read()

        for _, message in test_messages:
            assert message in log_content, f"Message '{message}' not found in log file"

        assert (
            "Blender_to_Spine2D_Mesh_Exporter" in log_content
        ), "Logger name not found in log output"
        assert "DEBUG" in log_content, "DEBUG level not found in log output"

    def test_logging_configuration_update_callback(self):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings
        mock_prefs.update_logging_config = MagicMock()

        addon_name = (
            Blender_to_Spine2D_Mesh_Exporter.__package__
            or "Blender_to_Spine2D_Mesh_Exporter"
        )
        bpy.context.preferences.addons[addon_name] = MagicMock()
        bpy.context.preferences.addons[addon_name].preferences = mock_prefs

        mock_context = MagicMock()
        mock_context.preferences.addons = bpy.context.preferences.addons

        config._update_logging_config(None, mock_context)

        mock_prefs.update_logging_config.assert_called_once()

    @patch("logging.config.dictConfig")
    def test_logging_level_hierarchy_integration(self, mock_dict_config):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings

        mock_logging_settings.enable_file_logging = False
        mock_logging_settings.modules = []

        module_hierarchy = [
            ("Blender_to_Spine2D_Mesh_Exporter", "ERROR"),
            ("config", "WARNING"),
            ("ui", "INFO"),
            ("main", "DEBUG"),
            ("texture_baker", "ERROR"),
            ("json_export", "INFO"),
        ]

        mock_modules = []
        for module_name, level in module_hierarchy:
            mock_module = MagicMock()
            mock_module.name = module_name
            mock_module.level = level
            mock_modules.append(mock_module)

        mock_logging_settings.modules = mock_modules

        addon_name = Blender_to_Spine2D_Mesh_Exporter.__name__
        bpy.context.preferences.addons[addon_name] = MagicMock()
        bpy.context.preferences.addons[addon_name].preferences = mock_prefs

        config.setup_logging()

        mock_dict_config.assert_called_once()
        config_dict = mock_dict_config.call_args[0][0]
        loggers = config_dict["loggers"]

        for module_name, expected_level in module_hierarchy:
            if module_name == "Blender_to_Spine2D_Mesh_Exporter":
                logger_key = "Blender_to_Spine2D_Mesh_Exporter"
            else:
                # Use the hardcoded addon ID from the manifest
                logger_key = f"bl_ext.user_default.Blender_to_Spine2D_Mesh_Exporter.{module_name}"

            assert logger_key in loggers, f"Logger {logger_key} not configured"
            assert (
                loggers[logger_key]["level"] == expected_level
            ), f"Logger {logger_key} has incorrect level"


class TestLoggingErrorRecoveryIntegration:
    def setup_method(self):
        bpy.reset_mock()
        bpy.context = MagicMock()
        bpy.context.preferences = MagicMock()
        bpy.context.preferences.addons = {}
        importlib.reload(config)

    @patch("logging.config.dictConfig")
    @patch.object(config, "_setup_default_logging")
    def test_fallback_to_default_logging_on_preferences_error(
        self, mock_default_logging, mock_dict_config
    ):
        addon_name = Blender_to_Spine2D_Mesh_Exporter.__name__
        # Simulate missing addon preferences
        bpy.context.preferences.addons = {}

        config.setup_logging()

        mock_default_logging.assert_called_once()

    @pytest.mark.skipif(
        sys.platform != "win32", reason="Windows-specific invalid path test"
    )
    @patch("logging.config.dictConfig")
    def test_invalid_log_file_path_handling(self, mock_dict_config):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings

        mock_logging_settings.enable_file_logging = True
        # Use a path with an invalid character to ensure failure on all platforms
        mock_logging_settings.log_file_path = "C:\\invalid:dir\\cannot_write.log"
        mock_logging_settings.modules = []

        addon_name = Blender_to_Spine2D_Mesh_Exporter.__name__
        bpy.context.preferences.addons[addon_name] = MagicMock()
        bpy.context.preferences.addons[addon_name].preferences = mock_prefs

        # Mock abspath to just return the invalid path
        bpy.path.abspath.side_effect = lambda p: p

        try:
            config.setup_logging()
        except Exception as e:
            pytest.fail(f"Logging setup should handle invalid paths gracefully: {e}")

        mock_dict_config.assert_called_once()

        config_dict = mock_dict_config.call_args[0][0]
        assert (
            "file" not in config_dict["handlers"]
        ), "File handler should not be added with invalid path"

    @patch("logging.config.dictConfig")
    @patch("builtins.print")
    def test_dictconfig_failure_recovery(self, mock_print, mock_dict_config):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings

        mock_logging_settings.enable_file_logging = False
        mock_logging_settings.modules = []

        addon_name = Blender_to_Spine2D_Mesh_Exporter.__name__
        bpy.context.preferences.addons[addon_name] = MagicMock()
        bpy.context.preferences.addons[addon_name].preferences = mock_prefs

        mock_dict_config.side_effect = Exception("Logging configuration failed")

        with patch.object(config, "_setup_default_logging") as mock_fallback:
            config.setup_logging()
            mock_fallback.assert_called_once()

        mock_print.assert_called()
        error_message = mock_print.call_args[0][0]
        assert "Error applying user logging config" in error_message


class TestLoggingPerformanceIntegration:
    def setup_method(self):
        bpy.reset_mock()
        bpy.context = MagicMock()
        self.temp_dir = tempfile.mkdtemp(
            prefix="Blender_to_Spine2D_Mesh_Exporter_perf_test_"
        )
        self.temp_log_file = os.path.join(self.temp_dir, "performance_test.log")
        bpy.path.abspath = lambda p: os.path.abspath(p)

    def teardown_method(self):
        import shutil

        _close_all_loggers()
        # Clean up logging handlers to avoid interference between tests
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        logging.shutdown()

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_high_volume_logging_performance(self):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings

        mock_logging_settings.enable_file_logging = True
        mock_logging_settings.log_file_path = self.temp_log_file

        mock_module = MagicMock()
        mock_module.name = "Blender_to_Spine2D_Mesh_Exporter"
        mock_module.level = "DEBUG"
        mock_logging_settings.modules = [mock_module]

        addon_name = Blender_to_Spine2D_Mesh_Exporter.__name__
        bpy.context.preferences.addons = {addon_name: MagicMock()}
        bpy.context.preferences.addons[addon_name].preferences = mock_prefs

        config.setup_logging()

        test_logger = logging.getLogger("Blender_to_Spine2D_Mesh_Exporter")
        message_count = 1000

        start_time = time.time()

        for i in range(message_count):
            test_logger.debug(f"Performance test message {i:04d}")
            test_logger.info(f"Info performance test message {i:04d}")
            if i % 100 == 0:
                test_logger.warning(f"Warning at message {i}")

        for handler in logging.getLogger().handlers:
            handler.flush()
            handler.close()

        elapsed_time = time.time() - start_time

        max_acceptable_time = 5.0
        assert (
            elapsed_time < max_acceptable_time
        ), f"High-volume logging took {elapsed_time:.2f}s (max: {max_acceptable_time}s)"

        assert os.path.exists(self.temp_log_file), "Log file was not created"

        with open(self.temp_log_file, "r", encoding="utf-8") as log_file:
            log_lines = log_file.readlines()

        expected_min_lines = message_count * 2
        assert (
            len(log_lines) >= expected_min_lines
        ), f"Expected at least {expected_min_lines} log lines, got {len(log_lines)}"

    def test_concurrent_logging_thread_safety(self):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings

        mock_logging_settings.enable_file_logging = True
        mock_logging_settings.log_file_path = self.temp_log_file

        mock_module = MagicMock()
        mock_module.name = "Blender_to_Spine2D_Mesh_Exporter"
        mock_module.level = "INFO"
        mock_logging_settings.modules = [mock_module]

        addon_name = Blender_to_Spine2D_Mesh_Exporter.__name__
        bpy.context.preferences.addons = {addon_name: MagicMock()}
        bpy.context.preferences.addons[addon_name].preferences = mock_prefs

        config.setup_logging()

        test_logger = logging.getLogger("Blender_to_Spine2D_Mesh_Exporter")
        thread_count = 5
        messages_per_thread = 100
        threads = []

        def logging_worker(thread_id):
            for i in range(messages_per_thread):
                test_logger.info(f"Thread {thread_id:02d} message {i:03d}")
                time.sleep(0.001)

        start_time = time.time()
        for thread_id in range(thread_count):
            thread = threading.Thread(target=logging_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=10.0)

        elapsed_time = time.time() - start_time

        # Close handlers to ensure file is written
        for handler in logging.getLogger().handlers:
            handler.flush()
            handler.close()

        max_acceptable_time = 15.0
        assert (
            elapsed_time < max_acceptable_time
        ), f"Concurrent logging took {elapsed_time:.2f}s (max: {max_acceptable_time}s)"

        assert os.path.exists(self.temp_log_file), "Log file was not created"

        with open(self.temp_log_file, "r", encoding="utf-8") as log_file:
            log_content = log_file.read()

        for thread_id in range(thread_count):
            thread_pattern = f"Thread {thread_id:02d}"
            assert (
                thread_pattern in log_content
            ), f"Messages from thread {thread_id} not found in log"

        total_expected_messages = thread_count * messages_per_thread
        log_lines = log_content.split("\n")
        actual_message_lines = [
            line for line in log_lines if "Thread" in line and "message" in line
        ]

        assert (
            len(actual_message_lines) == total_expected_messages
        ), f"Expected {total_expected_messages} messages, found {len(actual_message_lines)}"


class TestLoggingConfigurationEdgeCases:
    def setup_method(self):
        bpy.reset_mock()
        bpy.context = MagicMock()
        bpy.context.preferences = MagicMock()
        bpy.context.preferences.addons = {}
        importlib.reload(config)

    @patch("logging.config.dictConfig")
    def test_empty_modules_list_handling(self, mock_dict_config):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings

        mock_logging_settings.enable_file_logging = False
        mock_logging_settings.modules = []

        addon_name = Blender_to_Spine2D_Mesh_Exporter.__name__
        bpy.context.preferences.addons[addon_name] = MagicMock()
        bpy.context.preferences.addons[addon_name].preferences = mock_prefs

        config.setup_logging()

        mock_dict_config.assert_called_once()
        config_dict = mock_dict_config.call_args[0][0]

        assert "handlers" in config_dict
        assert "console" in config_dict["handlers"]

    @patch("logging.config.dictConfig")
    def test_malformed_module_settings_handling(self, mock_dict_config):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings

        mock_logging_settings.enable_file_logging = False

        malformed_module1 = MagicMock()
        malformed_module1.name = None
        malformed_module1.level = "DEBUG"

        malformed_module2 = MagicMock()
        malformed_module2.name = "valid_module"
        malformed_module2.level = "INVALID_LEVEL"

        mock_logging_settings.modules = [malformed_module1, malformed_module2]

        addon_name = Blender_to_Spine2D_Mesh_Exporter.__name__
        bpy.context.preferences.addons[addon_name] = MagicMock()
        bpy.context.preferences.addons[addon_name].preferences = mock_prefs

        try:
            config.setup_logging()
        except Exception as e:
            pytest.fail(f"Should handle malformed module settings gracefully: {e}")

        mock_dict_config.assert_called_once()

    def test_rapid_configuration_updates(self):
        mock_prefs = MagicMock()
        mock_logging_settings = MagicMock()
        mock_prefs.logging_settings = mock_logging_settings
        mock_prefs.update_logging_config = MagicMock()

        addon_name = (
            Blender_to_Spine2D_Mesh_Exporter.__package__
            or "Blender_to_Spine2D_Mesh_Exporter"
        )
        bpy.context.preferences.addons[addon_name] = MagicMock()
        bpy.context.preferences.addons[addon_name].preferences = mock_prefs

        mock_context = MagicMock()
        mock_context.preferences.addons = bpy.context.preferences.addons

        update_count = 50
        for i in range(update_count):
            config._update_logging_config(None, mock_context)

        assert mock_prefs.update_logging_config.call_count == update_count

        assert mock_prefs is not None
        assert mock_context is not None
