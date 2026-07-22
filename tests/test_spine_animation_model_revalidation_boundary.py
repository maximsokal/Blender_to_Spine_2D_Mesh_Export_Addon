from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "domain" / "spine"
CONTRACT = SPINE / "animation_model_contract.py"
SERIALIZER = SPINE / "serializer.py"
MODEL = SPINE / "model.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_revalidation_contract_is_the_only_private_model_bridge():
    source = read(CONTRACT)
    serializer_source = read(SERIALIZER)

    for name in (
        "_validate_event_definitions",
        "_validate_animation_event_timelines",
        "_validate_animation_draw_order_timelines",
        "_validate_animation_slot_attachment_timelines",
    ):
        assert name in source
        assert name not in serializer_source


def test_contract_revalidates_in_model_order_without_mutation():
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


def test_model_remains_unchanged_owner_of_construction_time_validation():
    source = read(MODEL)

    assert "def _validate_animation_event_timelines(" in source
    assert "def _validate_animation_draw_order_timelines(" in source
    assert "def _validate_animation_slot_attachment_timelines(" in source
    assert "animation_model_contract" not in source
