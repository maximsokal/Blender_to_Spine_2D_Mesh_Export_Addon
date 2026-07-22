from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "domain" / "spine"
SETUP_SLOT = SPINE / "setup_slot_contract.py"
COLOR = SPINE / "slot_color_timeline_contract.py"
DEFORM = SPINE / "deform_timeline_contract.py"
SEQUENCE = SPINE / "sequence_timeline_contract.py"
SERIALIZER = SPINE / "serializer.py"
MODEL = SPINE / "model.py"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_three_output_boundaries_delegate_slot_lookup_to_shared_index():
    setup_source = read(SETUP_SLOT)
    assert "class SetupSlotIndex:" in setup_source
    assert "def require(self, slot_name: object, *, path: str) -> int:" in setup_source

    for contract in (COLOR, DEFORM, SEQUENCE):
        source = read(contract)
        assert "SetupSlotIndex" in source
        assert "resolve_setup_slot_index" in source
        assert "setup_slot_index: SetupSlotIndex | None = None" in source
        assert "slot_index.require(slot_name, path=slot_path)" in source
        assert "known_slot_names" not in source
        assert "ambiguous_slot_names" not in source


def test_serializer_builds_and_passes_one_exact_index():
    source = read(SERIALIZER)
    to_dict_source = source[source.index("def to_dict(") :]

    assert to_dict_source.count("slot_names = tuple(slot.name for slot in document.slots)") == 1
    assert to_dict_source.count("SetupSlotIndex(slot_names)") == 1
    assert "setup_slot_index = SetupSlotIndex(slot_names)" in to_dict_source

    color_call = to_dict_source[to_dict_source.index("validate_animation_slot_color_timelines(") : to_dict_source.index("validate_animation_curves(")]
    deform_call = to_dict_source[to_dict_source.index("validate_animation_deform_timelines(") : to_dict_source.index("validate_animation_sequence_timelines(")]
    sequence_call = to_dict_source[to_dict_source.index("validate_animation_sequence_timelines(") : to_dict_source.index('data: dict[str, Any] = {')]
    for call in (color_call, deform_call, sequence_call):
        assert "slot_names=slot_names" in call
        assert "setup_slot_index=setup_slot_index" in call


def test_serializer_keeps_existing_boundary_order():
    source = read(SERIALIZER)
    body = source[source.index("def to_dict(") :]
    names = (
        "self._validator.validate_or_raise(document)",
        "validate_setup_linked_meshes(",
        "setup_slot_index = SetupSlotIndex(slot_names)",
        "validate_animation_slot_color_timelines(",
        "validate_animation_curves(",
        "validate_animation_deform_timelines(",
        "validate_animation_sequence_timelines(",
        'data: dict[str, Any] = {',
    )
    positions = [body.index(name) for name in names]
    assert positions == sorted(positions)


def test_model_level_early_validation_is_not_moved_in_this_slice():
    model_source = read(MODEL)
    assert "setup_slot_contract" not in model_source
    assert "def _validate_animation_draw_order_timelines(" in model_source
    assert "def _validate_animation_slot_attachment_timelines(" in model_source
