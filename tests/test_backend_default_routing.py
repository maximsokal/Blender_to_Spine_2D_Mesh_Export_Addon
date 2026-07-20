from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.single_object_operator import (
    DEFAULT_SINGLE_BACKEND,
    SINGLE_BACKEND_PROPERTY,
    resolve_single_backend,
)
from Blender_to_Spine2D_Mesh_Exporter.ui import (
    DEFAULT_MULTI_BACKEND,
    MULTI_BACKEND_PROPERTY,
    resolve_multi_backend,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (None, "REWRITE"),
        ("", "REWRITE"),
        ("unknown", "REWRITE"),
        ("rewrite", "REWRITE"),
        ("  rewrite  ", "REWRITE"),
        ("legacy", "LEGACY"),
        ("  LEGACY  ", "LEGACY"),
    ),
)
def test_single_backend_is_rewrite_by_default_and_legacy_only_when_explicit(
    value,
    expected,
):
    scene = SimpleNamespace()
    if value is not None:
        setattr(scene, SINGLE_BACKEND_PROPERTY, value)
    assert resolve_single_backend(scene) == expected
    assert DEFAULT_SINGLE_BACKEND == "REWRITE"


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (None, "REWRITE"),
        ("", "REWRITE"),
        ("unknown", "REWRITE"),
        ("rewrite", "REWRITE"),
        ("  rewrite  ", "REWRITE"),
        ("legacy", "LEGACY"),
        ("  LEGACY  ", "LEGACY"),
    ),
)
def test_multi_backend_is_rewrite_by_default_and_legacy_only_when_explicit(
    value,
    expected,
):
    scene = SimpleNamespace()
    if value is not None:
        setattr(scene, MULTI_BACKEND_PROPERTY, value)
    assert resolve_multi_backend(scene) == expected
    assert DEFAULT_MULTI_BACKEND == "REWRITE"
