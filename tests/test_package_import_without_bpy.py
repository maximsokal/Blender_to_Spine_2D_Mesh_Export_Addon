from pathlib import Path
import subprocess
import sys


def test_domain_and_parity_cli_import_without_bpy():
    repository_root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import sys

class BlockBpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "bpy" or fullname.startswith("bpy."):
            raise ModuleNotFoundError("blocked bpy for no-runtime import test", name=fullname)
        return None

sys.meta_path.insert(0, BlockBpy())
assert 'bpy' not in sys.modules
from Blender_to_Spine2D_Mesh_Exporter.domain.spine import A1ParitySettings
import Blender_to_Spine2D_Mesh_Exporter as addon
assert A1ParitySettings().absolute_tolerance == 1e-4
assert addon.register() is None
assert addon.unregister() is None
print('domain-import-ok')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )

    # Isolated mode omits cwd from sys.path, so insert it explicitly while still
    # preventing pytest's in-process fake bpy module from leaking into the child.
    if result.returncode != 0 and "No module named 'Blender_to_Spine2D_Mesh_Exporter'" in result.stderr:
        code_with_path = (
            f"import sys; sys.path.insert(0, {str(repository_root)!r});\n" + code
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code_with_path],
            cwd=repository_root,
            text=True,
            capture_output=True,
            check=False,
        )

    assert result.returncode == 0, result.stderr
    assert "domain-import-ok" in result.stdout
