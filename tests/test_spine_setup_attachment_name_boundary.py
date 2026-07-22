from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "domain" / "spine"
CONTRACT = SPINE / "setup_attachment_contract.py"
ANIMATION = SPINE / "animation_model_contract.py"
MODEL = SPINE / "model.py"
LINKED = SPINE / "linked_mesh_contract.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_contract_is_model_independent_and_avoids_import_cycle():
    source = read(CONTRACT)

    assert "from .model import" not in source
    assert "MeshAttachment" not in source
    assert "Skin" not in source
    assert "from collections.abc import Mapping" in source


def test_index_is_frozen_and_retains_exact_skin_attachment_tuple():
    source = read(CONTRACT)

    assert "@dataclass(frozen=True, slots=True)" in source
    assert "skin_attachments: tuple[Mapping[str, Mapping[str, Any]], ...]" in source
    assert "self.skin_attachments" in source
    assert "tuple(self.skin_attachments)" not in source


def test_index_owns_cross_skin_union_and_immutable_nested_values():
    source = read(CONTRACT)

    assert "mutable_names.setdefault(resolved_slot_name, set())" in source
    assert "names.add(" in source
    assert "frozenset(attachment_names)" in source
    assert "MappingProxyType(" in source
    assert "def names_for_slot(" in source


def test_index_owns_path_aware_attachment_reference_failure():
    source = read(CONTRACT)

    assert "def require(" in source
    assert 'raise ValueError("path must be a non-empty string")' in source
    assert "references undefined attachment" in source
    assert "for slot" in source


def test_model_delegates_attachment_validation_without_resolver_coupling():
    model_source = read(MODEL)
    animation_source = read(ANIMATION)

    assert (
        "from .animation_model_contract import validate_animation_model_contracts"
        in model_source
    )
    assert "setup_attachment_contract" not in model_source
    assert "SetupAttachmentNameIndex" not in model_source
    assert "linked_mesh_contract" not in model_source
    assert "LinkedMeshResolver" not in model_source

    assert "from .setup_attachment_contract import (" in animation_source
    assert "SetupAttachmentNameIndex" in animation_source
    assert "resolve_setup_attachment_name_index" in animation_source
    assert "linked_mesh_contract" not in animation_source
    assert "LinkedMeshResolver" not in animation_source


def test_linked_mesh_contract_keeps_exact_skin_specific_ownership():
    linked_source = read(LINKED)

    assert "class LinkedMeshResolver:" in linked_source
    assert "AttachmentReference" in linked_source
    assert "skin_name=skin.name" in linked_source
    assert "slot_name=slot_name" in linked_source
    assert "attachment_name=attachment_name" in linked_source
