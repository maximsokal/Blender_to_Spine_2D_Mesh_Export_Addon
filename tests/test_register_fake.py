# tests/test_register_fake.py
from unittest.mock import MagicMock, patch
import Blender_to_Spine2D_Mesh_Exporter

# Create a fake module with a register function
fake_module = MagicMock()
fake_module.register.return_value = None
fake_module.unregister.return_value = None


# Patch the list of modules used in __init__.py
@patch("Blender_to_Spine2D_Mesh_Exporter.MODULES", (fake_module,))
def test_register_unregister_no_errors():  # <-- FIX: Removed mock_modules argument
    """
    Checks that register/unregister call register/unregister
    on each module from the MODULES list.
    """
    # Reset call counters before the test
    fake_module.register.reset_mock()
    fake_module.unregister.reset_mock()

    Blender_to_Spine2D_Mesh_Exporter.register()
    # Check that the register method of our fake module was called
    fake_module.register.assert_called_once()

    Blender_to_Spine2D_Mesh_Exporter.unregister()
    # Check that the unregister method of our fake module was called
    fake_module.unregister.assert_called_once()
