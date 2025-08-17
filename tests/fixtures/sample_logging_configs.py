# tests/fixtures/sample_logging_configs.py
"""
Sample Logging Configuration Test Fixtures

This module provides standardized test data and configuration templates
for comprehensive testing of the addon's logging system. It includes
realistic configuration scenarios, edge cases, and performance test data.

## Configuration Categories:
1. Standard Production Configurations
2. Development and Debug Configurations
3. Edge Case and Error Scenarios
4. Performance Testing Configurations
5. Integration Test Templates

## Usage Pattern:
```python
from tests.fixtures.sample_logging_configs import PRODUCTION_CONFIG
mock_logging_settings.configure_from_dict(PRODUCTION_CONFIG)
```

Author: Test Fixture Collection
"""
import tempfile
from typing import Dict, List, NamedTuple
from unittest.mock import MagicMock


class ModuleLogConfig(NamedTuple):
    """Structured representation of module logging configuration."""

    name: str
    level: str
    description: str = ""


class LoggingTestConfig(NamedTuple):
    """Complete logging configuration for testing scenarios."""

    name: str
    description: str
    enable_file_logging: bool
    log_file_path: str
    modules: List[ModuleLogConfig]


# =============================================================================
# STANDARD PRODUCTION CONFIGURATIONS
# =============================================================================

PRODUCTION_CONFIG = LoggingTestConfig(
    name="production_standard",
    description="Standard production configuration with error-level logging",
    enable_file_logging=False,
    log_file_path="",
    modules=[
        ModuleLogConfig(
            "Blender_to_Spine2D_Mesh_Exporter", "ERROR", "Main addon logger"
        ),
        ModuleLogConfig("config", "ERROR", "Configuration module"),
        ModuleLogConfig("ui", "ERROR", "User interface module"),
        ModuleLogConfig("main", "ERROR", "Main processing module"),
        ModuleLogConfig("texture_baker", "ERROR", "Texture baking operations"),
        ModuleLogConfig("json_export", "ERROR", "JSON export functionality"),
        ModuleLogConfig("json_merger", "ERROR", "JSON merging operations"),
        ModuleLogConfig("plane_cut", "ERROR", "Mesh cutting operations"),
        ModuleLogConfig("seam_marker", "ERROR", "UV seam marking"),
        ModuleLogConfig("utils", "ERROR", "Utility functions"),
        ModuleLogConfig("uv_operations", "ERROR", "UV manipulation"),
        ModuleLogConfig("multi_object_export", "ERROR", "Multi-object processing"),
        ModuleLogConfig(
            "texture_baker_integration", "ERROR", "Texture baker integration"
        ),
    ],
)

PRODUCTION_WITH_FILE_LOGGING = LoggingTestConfig(
    name="production_file_logging",
    description="Production configuration with file logging enabled",
    enable_file_logging=True,
    log_file_path="/var/log/blender/Blender_to_Spine2D_Mesh_Exporter.log",
    modules=PRODUCTION_CONFIG.modules,
)

# =============================================================================
# DEVELOPMENT AND DEBUG CONFIGURATIONS
# =============================================================================

DEVELOPMENT_CONFIG = LoggingTestConfig(
    name="development_debug",
    description="Development configuration with verbose logging",
    enable_file_logging=True,
    log_file_path="/tmp/Blender_to_Spine2D_Mesh_Exporter.log",
    modules=[
        ModuleLogConfig(
            "Blender_to_Spine2D_Mesh_Exporter", "DEBUG", "Full debugging enabled"
        ),
        ModuleLogConfig("config", "DEBUG", "Configuration debugging"),
        ModuleLogConfig("ui", "INFO", "UI interaction logging"),
        ModuleLogConfig("main", "DEBUG", "Core logic debugging"),
        ModuleLogConfig("texture_baker", "INFO", "Texture operations tracking"),
        ModuleLogConfig("json_export", "DEBUG", "Export process debugging"),
        ModuleLogConfig("json_merger", "DEBUG", "Merge process debugging"),
        ModuleLogConfig("plane_cut", "INFO", "Cutting operation tracking"),
        ModuleLogConfig("seam_marker", "INFO", "Seam marking progress"),
        ModuleLogConfig("utils", "WARNING", "Utility warnings only"),
        ModuleLogConfig("uv_operations", "DEBUG", "UV manipulation debugging"),
        ModuleLogConfig("multi_object_export", "INFO", "Batch processing tracking"),
        ModuleLogConfig("texture_baker_integration", "DEBUG", "Integration debugging"),
    ],
)

SELECTIVE_DEBUG_CONFIG = LoggingTestConfig(
    name="selective_debug",
    description="Debug configuration for specific modules only",
    enable_file_logging=True,
    log_file_path="/tmp/Blender_to_Spine2D_Mesh_Exporter.log",
    modules=[
        ModuleLogConfig(
            "Blender_to_Spine2D_Mesh_Exporter", "INFO", "General information"
        ),
        ModuleLogConfig("config", "ERROR", "Config errors only"),
        ModuleLogConfig("ui", "WARNING", "UI warnings"),
        ModuleLogConfig("main", "DEBUG", "Core debugging enabled"),
        ModuleLogConfig("texture_baker", "DEBUG", "Texture debugging enabled"),
        ModuleLogConfig("json_export", "ERROR", "Export errors only"),
        ModuleLogConfig("json_merger", "ERROR", "Merge errors only"),
        ModuleLogConfig("plane_cut", "ERROR", "Cut errors only"),
        ModuleLogConfig("seam_marker", "ERROR", "Seam errors only"),
        ModuleLogConfig("utils", "ERROR", "Utility errors only"),
        ModuleLogConfig("uv_operations", "ERROR", "UV errors only"),
        ModuleLogConfig("multi_object_export", "INFO", "Export progress tracking"),
        ModuleLogConfig("texture_baker_integration", "DEBUG", "Integration debugging"),
    ],
)

# =============================================================================
# EDGE CASE AND ERROR SCENARIOS
# =============================================================================

EMPTY_MODULES_CONFIG = LoggingTestConfig(
    name="empty_modules",
    description="Configuration with no modules defined",
    enable_file_logging=False,
    log_file_path="",
    modules=[],
)

INVALID_LEVELS_CONFIG = LoggingTestConfig(
    name="invalid_levels",
    description="Configuration with invalid logging levels",
    enable_file_logging=False,
    log_file_path="",
    modules=[
        ModuleLogConfig(
            "Blender_to_Spine2D_Mesh_Exporter", "INVALID_LEVEL", "Invalid level test"
        ),
        ModuleLogConfig("config", "TRACE", "Non-existent level"),
        ModuleLogConfig("ui", "VERBOSE", "Custom level"),
        ModuleLogConfig("main", "", "Empty level"),
    ],
)

MALFORMED_MODULES_CONFIG = LoggingTestConfig(
    name="malformed_modules",
    description="Configuration with malformed module entries",
    enable_file_logging=False,
    log_file_path="",
    modules=[
        ModuleLogConfig("", "DEBUG", "Empty module name"),
        ModuleLogConfig("valid_module", "INFO", "Valid module for comparison"),
        ModuleLogConfig("module with spaces", "WARNING", "Invalid characters"),
        ModuleLogConfig(
            "very_long_module_name_that_exceeds_typical_limits_and_might_cause_issues",
            "ERROR",
            "Excessively long name",
        ),
    ],
)

INVALID_FILE_PATHS_CONFIG = LoggingTestConfig(
    name="invalid_file_paths",
    description="Configuration with problematic file paths",
    enable_file_logging=True,
    log_file_path="/root/protected/cannot_write.log",  # Typically unwritable
    modules=[
        ModuleLogConfig(
            "Blender_to_Spine2D_Mesh_Exporter", "DEBUG", "Testing with invalid path"
        ),
    ],
)

SPECIAL_CHARACTERS_CONFIG = LoggingTestConfig(
    name="special_characters",
    description="Configuration with special characters in paths",
    enable_file_logging=True,
    log_file_path="/tmp/spine2d_ñ漢字_тест.log",  # Unicode characters
    modules=[
        ModuleLogConfig(
            "Blender_to_Spine2D_Mesh_Exporter_ñ", "DEBUG", "Unicode in module name"
        ),
        ModuleLogConfig("config漢字", "INFO", "Asian characters"),
        ModuleLogConfig("ui_тест", "WARNING", "Cyrillic characters"),
    ],
)

# =============================================================================
# PERFORMANCE TESTING CONFIGURATIONS
# =============================================================================

HIGH_VOLUME_CONFIG = LoggingTestConfig(
    name="high_volume_testing",
    description="Configuration optimized for high-volume logging tests",
    enable_file_logging=True,
    log_file_path="/tmp/Blender_to_Spine2D_Mesh_Exporter_performance.log",
    modules=[
        ModuleLogConfig(
            "Blender_to_Spine2D_Mesh_Exporter",
            "DEBUG",
            "Full logging for performance test",
        ),
        ModuleLogConfig("config", "DEBUG", "Config performance testing"),
        ModuleLogConfig("ui", "DEBUG", "UI performance testing"),
        ModuleLogConfig("main", "DEBUG", "Main performance testing"),
        ModuleLogConfig("texture_baker", "DEBUG", "Texture performance testing"),
        ModuleLogConfig("json_export", "DEBUG", "Export performance testing"),
        ModuleLogConfig("json_merger", "DEBUG", "Merge performance testing"),
        ModuleLogConfig("plane_cut", "DEBUG", "Cut performance testing"),
        ModuleLogConfig("seam_marker", "DEBUG", "Seam performance testing"),
        ModuleLogConfig("utils", "DEBUG", "Utility performance testing"),
        ModuleLogConfig("uv_operations", "DEBUG", "UV performance testing"),
        ModuleLogConfig(
            "multi_object_export", "DEBUG", "Multi-object performance testing"
        ),
        ModuleLogConfig(
            "texture_baker_integration", "DEBUG", "Integration performance testing"
        ),
    ],
)

MINIMAL_LOGGING_CONFIG = LoggingTestConfig(
    name="minimal_logging",
    description="Minimal logging configuration for performance comparison",
    enable_file_logging=False,
    log_file_path="",
    modules=[
        ModuleLogConfig(
            "Blender_to_Spine2D_Mesh_Exporter", "CRITICAL", "Critical errors only"
        ),
    ],
)

# =============================================================================
# INTEGRATION TEST TEMPLATES
# =============================================================================

INTEGRATION_TEST_CONFIG = LoggingTestConfig(
    name="integration_testing",
    description="Balanced configuration for integration testing",
    enable_file_logging=True,
    log_file_path="/tmp/Blender_to_Spine2D_Mesh_Exporter_integration.log",
    modules=[
        ModuleLogConfig(
            "Blender_to_Spine2D_Mesh_Exporter", "INFO", "General integration logging"
        ),
        ModuleLogConfig("config", "WARNING", "Config integration warnings"),
        ModuleLogConfig("ui", "ERROR", "UI integration errors"),
        ModuleLogConfig("main", "INFO", "Main integration logging"),
        ModuleLogConfig("texture_baker", "WARNING", "Texture integration warnings"),
        ModuleLogConfig("json_export", "INFO", "Export integration logging"),
        ModuleLogConfig("json_merger", "INFO", "Merge integration logging"),
        ModuleLogConfig("plane_cut", "WARNING", "Cut integration warnings"),
        ModuleLogConfig("seam_marker", "WARNING", "Seam integration warnings"),
        ModuleLogConfig("utils", "ERROR", "Utility integration errors"),
        ModuleLogConfig("uv_operations", "WARNING", "UV integration warnings"),
        ModuleLogConfig(
            "multi_object_export", "INFO", "Multi-object integration logging"
        ),
        ModuleLogConfig("texture_baker_integration", "INFO", "Integration logging"),
    ],
)

# =============================================================================
# CONFIGURATION FACTORY FUNCTIONS
# =============================================================================


def create_temporary_log_path(prefix: str = "spine2d_test_") -> str:
    """
    Create a temporary file path for logging tests.

    Args:
        prefix: Filename prefix for the temporary log file

    Returns:
        str: Absolute path to temporary log file
    """
    temp_dir = tempfile.gettempdir()
    temp_file = tempfile.NamedTemporaryFile(
        prefix=prefix, suffix=".log", dir=temp_dir, delete=False
    )
    temp_file.close()
    return temp_file.name


def create_mock_preferences_from_config(config: LoggingTestConfig) -> MagicMock:
    """
    Create a mock Blender preferences object from a test configuration.

    This factory function generates properly structured mock objects
    that simulate Blender's addon preferences system with the specified
    logging configuration.

    Args:
        config: LoggingTestConfig object containing the desired configuration

    Returns:
        MagicMock: Configured mock object simulating Blender preferences

    Example:
        ```python
        mock_prefs = create_mock_preferences_from_config(PRODUCTION_CONFIG)
        # Use mock_prefs in test scenarios
        ```
    """
    mock_prefs = MagicMock()
    mock_logging_settings = MagicMock()

    # Configure main logging settings
    mock_logging_settings.enable_file_logging = config.enable_file_logging
    mock_logging_settings.log_file_path = config.log_file_path

    # Create mock modules
    mock_modules = []
    for module_config in config.modules:
        mock_module = MagicMock()
        mock_module.name = module_config.name
        mock_module.level = module_config.level
        mock_modules.append(mock_module)

    mock_logging_settings.modules = mock_modules
    mock_prefs.logging_settings = mock_logging_settings

    return mock_prefs


def get_all_test_configurations() -> Dict[str, LoggingTestConfig]:
    """
    Get all available test configurations as a dictionary.

    Returns:
        Dict[str, LoggingTestConfig]: Mapping of configuration names to objects

    Example:
        ```python
        configs = get_all_test_configurations()
        for name, config in configs.items():
            print(f"Testing configuration: {name}")
            test_logging_with_config(config)
        ```
    """
    return {
        "production_standard": PRODUCTION_CONFIG,
        "production_file_logging": PRODUCTION_WITH_FILE_LOGGING,
        "development_debug": DEVELOPMENT_CONFIG,
        "selective_debug": SELECTIVE_DEBUG_CONFIG,
        "empty_modules": EMPTY_MODULES_CONFIG,
        "invalid_levels": INVALID_LEVELS_CONFIG,
        "malformed_modules": MALFORMED_MODULES_CONFIG,
        "invalid_file_paths": INVALID_FILE_PATHS_CONFIG,
        "special_characters": SPECIAL_CHARACTERS_CONFIG,
        "high_volume": HIGH_VOLUME_CONFIG,
        "minimal_logging": MINIMAL_LOGGING_CONFIG,
        "integration_testing": INTEGRATION_TEST_CONFIG,
    }


def create_custom_config(
    name: str,
    description: str,
    enable_file_logging: bool = False,
    log_file_path: str = "",
    module_configs: List[tuple] = None,
) -> LoggingTestConfig:
    """
    Create a custom logging configuration for specific test scenarios.

    Args:
        name: Configuration identifier
        description: Human-readable description
        enable_file_logging: Whether to enable file logging
        log_file_path: Path to log file (if file logging enabled)
        module_configs: List of (name, level, description) tuples

    Returns:
        LoggingTestConfig: Custom configuration object

    Example:
        ```python
        custom_config = create_custom_config(
            name="test_scenario_1",
            description="Testing specific edge case",
            enable_file_logging=True,
            log_file_path="/tmp/test.log",
            module_configs=[
                ("Blender_to_Spine2D_Mesh_Exporter", "DEBUG", "Main debugging"),
                ("config", "INFO", "Config information")
            ]
        )
        ```
    """
    if module_configs is None:
        module_configs = [
            ("Blender_to_Spine2D_Mesh_Exporter", "ERROR", "Default configuration")
        ]

    modules = [
        ModuleLogConfig(name=name, level=level, description=desc)
        for name, level, desc in module_configs
    ]

    return LoggingTestConfig(
        name=name,
        description=description,
        enable_file_logging=enable_file_logging,
        log_file_path=log_file_path,
        modules=modules,
    )


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================


def validate_configuration(config: LoggingTestConfig) -> List[str]:
    """
    Validate a logging configuration and return any issues found.

    Args:
        config: Configuration to validate

    Returns:
        List[str]: List of validation issues (empty if valid)
    """
    issues = []

    # Validate main configuration
    if not config.name:
        issues.append("Configuration name cannot be empty")

    if config.enable_file_logging and not config.log_file_path:
        issues.append("File logging enabled but no log file path specified")

    # Validate modules
    if not config.modules:
        issues.append(
            "No modules configured (may be intentional for edge case testing)"
        )

    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

    for i, module in enumerate(config.modules):
        if not module.name:
            issues.append(f"Module {i}: Empty module name")

        if module.level not in valid_levels:
            issues.append(
                f"Module {i} ({module.name}): Invalid logging level '{module.level}'"
            )

    return issues


def get_recommended_configs_for_test_type(test_type: str) -> List[LoggingTestConfig]:
    """
    Get recommended configurations for specific types of tests.

    Args:
        test_type: Type of test ('unit', 'integration', 'performance', 'edge_case')

    Returns:
        List[LoggingTestConfig]: Recommended configurations for the test type
    """
    recommendations = {
        "unit": [PRODUCTION_CONFIG, DEVELOPMENT_CONFIG],
        "integration": [INTEGRATION_TEST_CONFIG, PRODUCTION_WITH_FILE_LOGGING],
        "performance": [HIGH_VOLUME_CONFIG, MINIMAL_LOGGING_CONFIG],
        "edge_case": [
            EMPTY_MODULES_CONFIG,
            INVALID_LEVELS_CONFIG,
            MALFORMED_MODULES_CONFIG,
            INVALID_FILE_PATHS_CONFIG,
            SPECIAL_CHARACTERS_CONFIG,
        ],
    }

    return recommendations.get(test_type, [PRODUCTION_CONFIG])
