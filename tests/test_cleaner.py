# tests/test_cleaner.py
import libcst as cst
from tools.prepare_package import (
    BlenderCSTCleaner,
)  # Assumes `tools` is in sys.path
from libcst.metadata import MetadataWrapper
from tools.prepare_package import BlenderCSTCleaner


def run_cleaner(source_code):
    tree = cst.parse_module(source_code)
    # FIX: Wrap the tree in MetadataWrapper before visiting
    wrapper = MetadataWrapper(tree)
    context = cst.codemod.CodemodContext()
    cleaner = BlenderCSTCleaner(context)
    modified_tree = wrapper.visit(cleaner)
    return modified_tree.code


def test_removes_print_statements():
    source = "def my_func():\n    print('debug')\n    return 1"
    expected = "def my_func():\n    return 1"
    assert run_cleaner(source).strip() == expected.strip()


def test_keeps_operator_docstring():
    source = '''
import bpy
class MY_OT_Operator(bpy.types.Operator):
    """This docstring must be kept."""
    def execute(self, context):
        return {'FINISHED'}
'''
    cleaned = run_cleaner(source)
    assert '"""This docstring must be kept."""' in cleaned
