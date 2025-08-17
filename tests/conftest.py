# tests/conftest.py
"""
Comprehensive Test Configuration and Fixture Management

This module provides the foundational testing infrastructure for the Spine2D Mesh Exporter addon.
It establishes a comprehensive set of fixtures, mocks, and utilities that ensure consistent,
reliable testing across all addon components.

## Core Testing Philosophy
The testing infrastructure follows these principles:
1. **Isolation**: Each test runs in a clean, predictable environment
2. **Realism**: Mocks accurately simulate Blender's behavior patterns
3. **Flexibility**: Fixtures can be customized for specific test scenarios
4. **Performance**: Efficient setup/teardown for rapid test execution

## Fixture Categories
- **Blender Environment Simulation**: Complete Blender API mocking
- **Logging System Fixtures**: Temporary logging configurations and file handling
- **Addon State Management**: Registration and preference state control
- **Data Generation**: Realistic test data for various scenarios

"""
import sys
import os
import tempfile
from unittest.mock import MagicMock
from pathlib import Path
import pytest

# Import test fixtures and sample configurations using absolute paths
try:
    # Primary import method - when run as package
    from tests.fixtures.sample_logging_configs import (
        # LoggingTestConfig,
        create_mock_preferences_from_config,
        # create_temporary_log_path,
        PRODUCTION_CONFIG,
        # DEVELOPMENT_CONFIG,
        INTEGRATION_TEST_CONFIG,
    )
except ImportError:
    # Fallback import method - when run from project root
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fixtures"))
    from sample_logging_configs import (
        create_mock_preferences_from_config,
        PRODUCTION_CONFIG,
        INTEGRATION_TEST_CONFIG,
    )


# =============================================================================
# ENHANCED BLENDER API SIMULATION
# =============================================================================


class EnhancedFakeOperator:
    """
    Advanced Blender Operator simulation with realistic behavior patterns.

    This class provides a more sophisticated mock of Blender's Operator system,
    including proper return value handling, context validation, and error simulation.
    """

    def __init__(self, simulate_failures: bool = False):
        """
        Initialize operator with configurable failure simulation.

        Args:
            simulate_failures: Whether to simulate random operator failures
        """
        self.simulate_failures = simulate_failures
        self.execution_count = 0
        self.last_context = None
        self.reported_messages = []

    def execute(self, context):
        """
        Simulate operator execution with realistic return patterns.

        Args:
            context: Mock Blender context object

        Returns:
            set: Blender operator return status
        """
        self.execution_count += 1
        self.last_context = context

        if self.simulate_failures and self.execution_count % 3 == 0:
            return {"CANCELLED"}

        return {"FINISHED"}

    @classmethod
    def poll(cls, context):
        """
        Simulate operator poll method with context validation.

        Args:
            context: Mock Blender context object

        Returns:
            bool: Whether operator can execute in current context
        """
        # Simulate realistic poll behavior
        return hasattr(context, "active_object") and context.active_object is not None

    def report(self, message_type: set, message: str):
        """
        Simulate Blender's operator message reporting system.

        Args:
            message_type: Set containing message type (e.g., {'INFO'}, {'ERROR'})
            message: Human-readable message string
        """
        self.reported_messages.append((message_type, message))

    def get_last_report(self) -> tuple:
        """Get the most recent report message."""
        return self.reported_messages[-1] if self.reported_messages else None


class EnhancedFakePanel:
    """
    Sophisticated Blender Panel simulation with layout management.

    Provides realistic simulation of Blender's UI panel system including
    layout operations and property display.
    """

    # Required Blender panel attributes
    bl_idname = "TEST_PT_panel"
    bl_label = "Test Panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Tools"

    def __init__(self):
        """Initialize panel with layout tracking."""
        self.draw_calls = []
        self.layout_operations = []

    def draw(self, context):
        """
        Simulate panel drawing with layout operation tracking.

        Args:
            context: Mock Blender context object
        """
        self.draw_calls.append(context)

        # Simulate common layout operations
        layout = EnhancedMockLayout()
        self.layout_operations.append(layout)
        return layout


class EnhancedMockLayout:
    """
    Advanced mock of Blender's UILayout system.

    Tracks UI operations for testing panel drawing logic.
    """

    def __init__(self):
        """Initialize layout with operation tracking."""
        self.operations = []
        self.props_displayed = []
        self.operators_added = []

    def prop(self, data, property_name, **kwargs):
        """Simulate property display in UI."""
        self.props_displayed.append((data, property_name, kwargs))
        return MagicMock()

    def operator(self, operator_id, **kwargs):
        """Simulate operator button in UI."""
        self.operators_added.append((operator_id, kwargs))
        return MagicMock()

    def label(self, text="", **kwargs):
        """Simulate text label in UI."""
        self.operations.append(("label", text, kwargs))
        return MagicMock()

    def box(self):
        """Simulate UI box container."""
        box = EnhancedMockLayout()
        self.operations.append(("box", box))
        return box

    def column(self, align=False):
        """Simulate UI column layout."""
        column = EnhancedMockLayout()
        self.operations.append(("column", column, {"align": align}))
        return column

    def row(self, align=False):
        """Simulate UI row layout."""
        row = EnhancedMockLayout()
        self.operations.append(("row", row, {"align": align}))
        return row


class EnhancedFakeVector:
    """
    Comprehensive Vector simulation with mathematical operations.

    Provides realistic 3D vector operations for geometric calculations
    commonly used in mesh processing operations.
    """

    def __init__(self, values=(0.0, 0.0, 0.0)):
        """
        Initialize vector with coordinate values.

        Args:
            values: Tuple of (x, y, z) coordinate values
        """
        self.values = tuple(float(v) for v in values)
        self.x, self.y, self.z = self.values
        self._update_computed_properties()

    def _update_computed_properties(self):
        """Update computed properties after value changes."""
        self.length = (sum(v * v for v in self.values)) ** 0.5
        self.length_squared = sum(v * v for v in self.values)

    def __sub__(self, other):
        """Vector subtraction operation."""
        if isinstance(other, EnhancedFakeVector):
            return EnhancedFakeVector(
                [a - b for a, b in zip(self.values, other.values)]
            )
        return EnhancedFakeVector([a - other for a in self.values])

    def __add__(self, other):
        """Vector addition operation."""
        if isinstance(other, EnhancedFakeVector):
            return EnhancedFakeVector(
                [a + b for a, b in zip(self.values, other.values)]
            )
        return EnhancedFakeVector([a + other for a in self.values])

    def __mul__(self, scalar):
        """Vector scalar multiplication."""
        return EnhancedFakeVector([v * scalar for v in self.values])

    def __rmul__(self, scalar):
        """Reverse scalar multiplication."""
        return self.__mul__(scalar)

    def dot(self, other):
        """Vector dot product calculation."""
        return sum(a * b for a, b in zip(self.values, other.values))

    def cross(self, other):
        """Vector cross product calculation."""
        x = self.y * other.z - self.z * other.y
        y = self.z * other.x - self.x * other.z
        z = self.x * other.y - self.y * other.x
        return EnhancedFakeVector([x, y, z])

    def normalized(self):
        """Return normalized vector."""
        if self.length == 0:
            return EnhancedFakeVector([0, 0, 0])
        return EnhancedFakeVector([v / self.length for v in self.values])

    def copy(self):
        """Create a copy of the vector."""
        return EnhancedFakeVector(self.values)

    def __repr__(self):
        """String representation for debugging."""
        return f"Vector(({self.x:.3f}, {self.y:.3f}, {self.z:.3f}))"


# =============================================================================
# CORE BLENDER API MOCKS
# =============================================================================


def create_enhanced_bpy_mock():
    """
    Create comprehensive Blender Python API mock with realistic behavior.

    Returns:
        MagicMock: Enhanced bpy module mock with full API simulation
    """
    bpy_mock = MagicMock()

    # Configure types module
    bpy_mock.types.Operator = EnhancedFakeOperator
    bpy_mock.types.Panel = EnhancedFakePanel
    bpy_mock.types.PropertyGroup = MagicMock
    bpy_mock.types.AddonPreferences = MagicMock

    # Configure utilities
    bpy_mock.utils.register_class = MagicMock()
    bpy_mock.utils.unregister_class = MagicMock()

    # Configure props module
    bpy_mock.props.StringProperty = MagicMock(return_value=MagicMock())
    bpy_mock.props.BoolProperty = MagicMock(return_value=MagicMock())
    bpy_mock.props.FloatProperty = MagicMock(return_value=MagicMock())
    bpy_mock.props.EnumProperty = MagicMock(return_value=MagicMock())
    bpy_mock.props.PointerProperty = MagicMock(return_value=MagicMock())
    bpy_mock.props.CollectionProperty = MagicMock(return_value=MagicMock())

    # Configure operators
    bpy_mock.ops = MagicMock()
    bpy_mock.ops.preferences = MagicMock()
    bpy_mock.ops.preferences.addon_disable = MagicMock(return_value={"FINISHED"})
    bpy_mock.ops.preferences.addon_remove = MagicMock(return_value={"FINISHED"})

    # Configure context
    bpy_mock.context = MagicMock()
    bpy_mock.context.scene = MagicMock()
    bpy_mock.context.preferences = MagicMock()
    bpy_mock.context.preferences.addons = {}

    # Configure data access
    bpy_mock.data = MagicMock()
    bpy_mock.data.filepath = ""
    bpy_mock.data.is_saved = False

    # Configure path utilities
    bpy_mock.path = MagicMock()
    bpy_mock.path.abspath = MagicMock(side_effect=lambda x: x)

    return bpy_mock


def create_enhanced_mathutils_mock():
    """
    Create comprehensive mathutils module mock.

    Returns:
        MagicMock: Enhanced mathutils module with Vector support
    """
    mathutils_mock = MagicMock()
    mathutils_mock.Vector = EnhancedFakeVector
    return mathutils_mock


# =============================================================================
# PYTEST FIXTURES
# =============================================================================


@pytest.fixture(scope="function")
def clean_bpy_environment():
    """
    Provide clean Blender environment for each test.

    This fixture ensures each test starts with a fresh, predictable
    Blender API simulation state.

    Yields:
        MagicMock: Clean bpy module mock
    """
    # Create fresh mocks
    bpy_mock = create_enhanced_bpy_mock()
    mathutils_mock = create_enhanced_mathutils_mock()

    # Store original modules (if any)
    original_bpy = sys.modules.get("bpy")
    original_mathutils = sys.modules.get("mathutils")
    original_bmesh = sys.modules.get("bmesh")

    # Install mocks
    sys.modules["bpy"] = bpy_mock
    sys.modules["mathutils"] = mathutils_mock
    sys.modules["bmesh"] = MagicMock()

    try:
        yield bpy_mock
    finally:
        # Restore original modules
        if original_bpy is not None:
            sys.modules["bpy"] = original_bpy
        else:
            sys.modules.pop("bpy", None)

        if original_mathutils is not None:
            sys.modules["mathutils"] = original_mathutils
        else:
            sys.modules.pop("mathutils", None)

        if original_bmesh is not None:
            sys.modules["bmesh"] = original_bmesh
        else:
            sys.modules.pop("bmesh", None)


@pytest.fixture(scope="function")
def temporary_log_directory():
    """
    Provide temporary directory for logging tests.

    Creates a clean temporary directory for each test that requires
    file-based logging operations.

    Yields:
        Path: Temporary directory path
    """
    temp_dir = tempfile.mkdtemp(prefix="spine2d_test_")
    temp_path = Path(temp_dir)

    try:
        yield temp_path
    finally:
        # Clean up temporary directory
        import shutil

        if temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture(scope="function")
def mock_logging_preferences():
    """
    Provide mock logging preferences for testing.

    Creates a realistic mock of Blender's addon preferences system
    specifically configured for logging functionality testing.

    Yields:
        MagicMock: Configured logging preferences mock
    """
    mock_prefs = create_mock_preferences_from_config(PRODUCTION_CONFIG)

    # Add update_logging_config method
    mock_prefs.update_logging_config = MagicMock()

    yield mock_prefs


@pytest.fixture(scope="function")
def sample_logging_configs():
    """
    Provide collection of sample logging configurations.

    Returns a dictionary of various logging configurations for
    comprehensive testing scenarios.

    Returns:
        Dict[str, LoggingTestConfig]: Available test configurations
    """
    from .fixtures.sample_logging_configs import get_all_test_configurations

    return get_all_test_configurations()


@pytest.fixture(scope="function")
def temporary_log_file(temporary_log_directory):
    """
    Provide temporary log file for file logging tests.

    Args:
        temporary_log_directory: Temporary directory fixture

    Yields:
        str: Path to temporary log file
    """
    log_file_path = temporary_log_directory / "test_addon.log"
    yield str(log_file_path)

    # Cleanup handled by temporary_log_directory fixture


@pytest.fixture(scope="function")
def mock_addon_registration_system(clean_bpy_environment):
    """
    Provide mock addon registration system for testing.

    Creates a comprehensive mock of Blender's addon registration
    system including preferences management.

    Args:
        clean_bpy_environment: Clean Blender environment fixture

    Yields:
        Dict[str, Any]: Registration system components
    """
    bpy = clean_bpy_environment

    # Configure addon preferences structure
    addon_name = "test_addon"
    mock_preferences = MagicMock()
    mock_logging_settings = MagicMock()
    mock_preferences.logging_settings = mock_logging_settings

    # Configure preferences in context
    bpy.context.preferences.addons[addon_name] = MagicMock()
    bpy.context.preferences.addons[addon_name].preferences = mock_preferences

    registration_components = {
        "bpy": bpy,
        "addon_name": addon_name,
        "preferences": mock_preferences,
        "logging_settings": mock_logging_settings,
    }

    yield registration_components


@pytest.fixture(scope="function")
def logging_capture_handler():
    """
    Provide logging handler that captures log messages for testing.

    Creates a custom logging handler that stores log messages
    for verification in tests.

    Yields:
        logging.Handler: Custom handler with captured messages
    """
    import logging

    class TestLogHandler(logging.Handler):
        """Custom handler for capturing log messages."""

        def __init__(self):
            super().__init__()
            self.messages = []
            self.records = []

        def emit(self, record):
            """Capture log record."""
            self.records.append(record)
            self.messages.append(self.format(record))

        def get_messages_by_level(self, level):
            """Get messages filtered by logging level."""
            return [
                msg
                for record, msg in zip(self.records, self.messages)
                if record.levelno >= getattr(logging, level)
            ]

        def clear(self):
            """Clear captured messages."""
            self.messages.clear()
            self.records.clear()

    handler = TestLogHandler()

    # Configure formatter
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)

    # Add to root logger temporarily
    root_logger = logging.getLogger()
    original_level = root_logger.level
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    try:
        yield handler
    finally:
        # Cleanup
        root_logger.removeHandler(handler)
        root_logger.setLevel(original_level)


@pytest.fixture(scope="session")
def addon_test_constants():
    """
    Provide constants used across addon testing.

    Returns:
        Dict[str, Any]: Test constants and configuration values
    """
    return {
        "MODULE_NAMES": [
            "Blender_to_Spine2D_Mesh_Exporter",
            "config",
            "ui",
            "main",
            "texture_baker",
            "json_export",
            "json_merger",
            "plane_cut",
            "seam_marker",
            "utils",
            "uv_operations",
            "multi_object_export",
            "texture_baker_integration",
        ],
        "LOG_LEVELS": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "DEFAULT_LOG_LEVEL": "ERROR",
        "TEST_TIMEOUT": 30.0,
        "PERFORMANCE_THRESHOLD": 5.0,
    }


# =============================================================================
# SPECIALIZED FIXTURES FOR INTEGRATION TESTING
# =============================================================================


@pytest.fixture(scope="function")
def integration_test_environment(clean_bpy_environment, temporary_log_directory):
    """
    Provide complete environment for integration testing.

    Combines multiple fixtures to create a comprehensive testing
    environment suitable for complex integration scenarios.

    Args:
        clean_bpy_environment: Clean Blender environment fixture
        temporary_log_directory: Temporary directory fixture

    Yields:
        Dict[str, Any]: Complete integration test environment
    """
    bpy = clean_bpy_environment

    # Configure realistic addon structure
    addon_name = "Blender_to_Spine2D_Mesh_Exporter"

    # Create comprehensive preferences mock
    mock_prefs = create_mock_preferences_from_config(INTEGRATION_TEST_CONFIG)
    mock_prefs.update_logging_config = MagicMock()

    # Update log file path to use temporary directory
    log_file_path = temporary_log_directory / "integration_test.log"
    mock_prefs.logging_settings.log_file_path = str(log_file_path)

    # Configure in Blender context
    bpy.context.preferences.addons[addon_name] = MagicMock()
    bpy.context.preferences.addons[addon_name].preferences = mock_prefs

    # Configure path operations
    bpy.path.abspath.side_effect = lambda x: str(Path(x).resolve())

    integration_environment = {
        "bpy": bpy,
        "addon_name": addon_name,
        "preferences": mock_prefs,
        "log_directory": temporary_log_directory,
        "log_file_path": log_file_path,
        "config": INTEGRATION_TEST_CONFIG,
    }

    yield integration_environment


# =============================================================================
# AUTOMATIC MOCK SETUP (LEGACY COMPATIBILITY)
# =============================================================================

# Apply global mocks for legacy compatibility
# These are automatically applied when conftest.py is imported

# Create and install enhanced mocks
bpy_mock = create_enhanced_bpy_mock()
mathutils_mock = create_enhanced_mathutils_mock()

# Install global mocks (preserved for backwards compatibility)
sys.modules["bpy"] = bpy_mock
sys.modules["mathutils"] = mathutils_mock
sys.modules["bmesh"] = MagicMock()


# =============================================================================
# PYTEST CONFIGURATION HOOKS
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "logging: mark test as logging-related test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers."""
    for item in items:
        # Auto-mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)

        # Auto-mark performance tests
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)

        # Auto-mark logging tests
        if "logging" in item.nodeid.lower():
            item.add_marker(pytest.mark.logging)
