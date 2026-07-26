from types import SimpleNamespace

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.image_dependency_preflight import (
    ImageDependencyPreflightError,
    preflight_object_image_dependencies,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    ImageDependency,
    MaterialAnalysis,
    MaterialKind,
    ObjectMaterialAnalysis,
)


class _Tree:
    def __init__(self, nodes):
        self.nodes = tuple(nodes)

    def as_pointer(self):
        return id(self)


class _Image:
    def __init__(self, name, path=None, *, source="FILE", packed=False):
        self.name = name
        self.name_full = name
        self.source = source
        self.filepath = "" if path is None else str(path)
        self.filepath_raw = "" if path is None else str(path)
        self.frame_duration = 1
        self.packed_file = object() if packed else None
        self.packed_files = ()
        self.library = None


class _ImageNode:
    type = "TEX_IMAGE"

    def __init__(self, image):
        self.image = image
        self.node_tree = None


class _Object:
    def __init__(self, image):
        material = SimpleNamespace(node_tree=_Tree((_ImageNode(image),)))
        self.material_slots = (SimpleNamespace(material=material),)


def _analysis(image_name, path=None, *, source="FILE"):
    dependency = ImageDependency(
        image_name=image_name,
        source=source,
        filepath=None if path is None else str(path),
        frame_duration=1,
        generated=source == "GENERATED",
    )
    return ObjectMaterialAnalysis(
        source_object_id="Object",
        slots=(
            MaterialAnalysis(
                slot_index=0,
                material_name="Material",
                kind=MaterialKind.IMAGE,
                node_types=("TEX_IMAGE",),
                image_dependencies=(dependency,),
            ),
        ),
    )


def test_missing_reachable_image_is_blocker(tmp_path):
    path = tmp_path / "missing.png"
    image = _Image("Missing", path)

    with pytest.raises(ImageDependencyPreflightError, match="Relink or pack"):
        preflight_object_image_dependencies(
            _Object(image),
            _analysis("Missing", path),
        )


def test_pathless_file_image_is_blocker():
    image = _Image("Pathless")

    with pytest.raises(ImageDependencyPreflightError, match="no resolvable file path"):
        preflight_object_image_dependencies(
            _Object(image),
            _analysis("Pathless"),
        )


def test_existing_reachable_image_passes(tmp_path):
    path = tmp_path / "texture.png"
    path.write_bytes(b"not decoded by preflight")
    image = _Image("Texture", path)

    assert preflight_object_image_dependencies(
        _Object(image),
        _analysis("Texture", path),
    ) == ("Texture",)


def test_packed_reachable_image_does_not_require_external_file(tmp_path):
    path = tmp_path / "missing-packed.png"
    image = _Image("Packed", path, packed=True)

    assert preflight_object_image_dependencies(
        _Object(image),
        _analysis("Packed", path),
    ) == ("Packed",)


def test_generated_reachable_image_does_not_require_external_file():
    image = _Image("Generated", source="GENERATED")

    assert preflight_object_image_dependencies(
        _Object(image),
        _analysis("Generated", source="GENERATED"),
    ) == ("Generated",)
