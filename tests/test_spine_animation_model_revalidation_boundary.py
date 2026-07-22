from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "domain" / "spine"
CONTRACT = SPINE / "animation_model_contract.py"
SERIALIZER = SPINE / "serializer.py"
MODEL = SPINE / "model.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_contract_owns_model_animation_implementation_without_import_cycle():
    source = read(CONTRACT)
    model_source = read(MODEL)
    serializer_source = read(SERIALIZER)

    assert "from .model import" not in source
    for name in (
        "_validate_event_definitions",
        "_validate_animation_event_timelines",
        "_validate_animation_draw_order_timelines",
        "_validate_animation_slot_attachment_timelines",
    ):
        assert f"def {name}(" in source
        assert f"def {name}(" not in model_source
        assert name not in serializer_source


def test_contract_validates_in_original_model_order_without_mutation():
    source = read(CONTRACT)
    names = (
        "validate_json_mapping(animations, path=path)",
        "validate_json_mapping(events, path=events_path)",
        "_validate_event_definitions(events, path=events_path)",
        "_validate_animation_event_timelines(",
        "slot_index = resolve_setup_slot_index(",
        "_validate_animation_draw_order_timelines(",
        "attachment_index = resolve_setup_attachment_name_index(",
        "_validate_animation_slot_attachment_timelines(",
    )
    positions = [source.index(name) for name in names]

    assert positions == sorted(positions)
    assert "setdefault(" not in source
    assert "deepcopy(" not in source


def test_contract_accepts_and_reuses_both_exact_indexes():
    source = read(CONTRACT)

    assert "setup_slot_index: SetupSlotIndex | None = None" in source
    assert "setup_attachment_index: SetupAttachmentNameIndex | None = None" in source
    assert "resolve_setup_slot_index(slot_names, setup_slot_index)" in source
    assert "setup_attachment_index," in source
    assert "setup_slot_index=slot_index" in source
    assert "setup_attachment_index=attachment_index" in source


def test_model_delegates_construction_validation_to_public_contract():
    source = read(MODEL)
    document_source = source[source.index("class SpineDocument:") :]
    call = document_source[
        document_source.index("validate_animation_model_contracts(") :
        document_source.index("_validate_extras(", document_source.index("validate_animation_model_contracts("))
    ]

    assert (
        "from .animation_model_contract import validate_animation_model_contracts"
        in source
    )
    assert "slot_names = tuple(slot.name for slot in self.slots)" in document_source
    assert "setup_slot_index = SetupSlotIndex(slot_names)" in document_source
    assert "skin_attachments = tuple(skin.attachments for skin in self.skins)" in document_source
    assert "events=self.events" in call
    assert "slot_names=slot_names" in call
    assert "skin_attachments=skin_attachments" in call
    assert "setup_slot_index=setup_slot_index" in call
    assert "setup_attachment_index=" not in call


def test_serializer_revalidates_before_output_specific_boundaries():
    source = read(SERIALIZER)
    body = source[source.index("def to_dict(") :]
    names = (
        "self._validator.validate_or_raise(document)",
        "setup_slot_index = SetupSlotIndex(slot_names)",
        "setup_attachment_index = SetupAttachmentNameIndex(skin_attachments)",
        "validate_animation_model_contracts(",
        "validate_setup_linked_meshes(",
        "validate_animation_slot_color_timelines(",
        "validate_animation_curves(",
        "validate_animation_deform_timelines(",
        "validate_animation_sequence_timelines(",
        'data: dict[str, Any] = {',
    )
    positions = [body.index(name) for name in names]

    assert positions == sorted(positions)


def test_serializer_passes_exact_snapshots_and_indexes_to_revalidation():
    source = read(SERIALIZER)
    body = source[source.index("def to_dict(") :]
    call = body[
        body.index("validate_animation_model_contracts(") :
        body.index("validate_setup_linked_meshes(")
    ]

    assert "events=document.events" in call
    assert "slot_names=slot_names" in call
    assert "skin_attachments=skin_attachments" in call
    assert "setup_slot_index=setup_slot_index" in call
    assert "setup_attachment_index=setup_attachment_index" in call
