from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPINE = ROOT / "Blender_to_Spine2D_Mesh_Exporter" / "domain" / "spine"
CONTRACT = SPINE / "animation_model_contract.py"
MODEL = SPINE / "model.py"
SERIALIZER = SPINE / "serializer.py"
GROUPED_OVERLAY = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "application"
    / "a1_grouped_camera_projection.py"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def slot_attachment_helper_source() -> str:
    source = read(CONTRACT)
    helper_start = source.index(
        "def _validate_animation_slot_attachment_timelines("
    )
    next_helper = source.index(
        "def validate_animation_model_contracts(",
        helper_start,
    )
    return source[helper_start:next_helper]


def test_slot_attachment_validation_runs_after_event_and_draw_order_contracts():
    source = read(CONTRACT)
    body = source[source.index("def validate_animation_model_contracts(") :]

    json_index = body.index("validate_json_mapping(animations, path=path)")
    event_index = body.index("_validate_animation_event_timelines(")
    draw_order_index = body.index("_validate_animation_draw_order_timelines(")
    attachment_index = body.index("_validate_animation_slot_attachment_timelines(")

    assert json_index < event_index < draw_order_index < attachment_index


def test_model_passes_cross_skin_snapshot_to_public_contract():
    source = read(MODEL)
    document_source = source[source.index("class SpineDocument:") :]
    call = document_source[
        document_source.index("validate_animation_model_contracts(") :
        document_source.index("_validate_extras(", document_source.index("validate_animation_model_contracts("))
    ]

    assert "skin_attachments = tuple(skin.attachments for skin in self.skins)" in document_source
    assert "skin_attachments=skin_attachments" in call
    assert "setup_attachment_index=" not in call
    assert "SetupAttachmentNameIndex" not in source
    assert "attachment_names_by_slot: dict[str, set[str]] = {}" not in source
    assert "attachment_names_by_slot.setdefault(" not in source
    assert "self.skins[0]" not in document_source
    assert 'skin.name == "default"' not in document_source


def test_slot_attachment_containers_are_fail_closed():
    helper = slot_attachment_helper_source()

    assert 'if "slots" not in animation_metadata:' in helper
    assert "if not isinstance(slot_timelines, Mapping):" in helper
    assert "if not isinstance(slot_metadata, Mapping):" in helper
    assert 'if "attachment" not in slot_metadata:' in helper
    assert "if not isinstance(timeline, (list, tuple)):" in helper
    assert "if not timeline:" in helper
    assert "if not isinstance(keyframe, Mapping):" in helper


def test_slot_references_use_shared_exact_index():
    helper = slot_attachment_helper_source()

    assert "setup_slot_index: SetupSlotIndex" in helper
    assert "isinstance(setup_slot_index, SetupSlotIndex)" in helper
    assert "slot_path = json_path_key(slots_path, slot_name)" in helper
    assert '_require_name(slot_name, f"{slot_path} slot name")' in helper
    assert "setup_slot_index.require(slot_name, path=slot_path)" in helper
    assert "known_slot_names" not in helper
    assert "ambiguous_slot_names" not in helper


def test_attachment_names_use_shared_cross_skin_index():
    helper = slot_attachment_helper_source()

    assert "setup_attachment_index: SetupAttachmentNameIndex" in helper
    assert "isinstance(setup_attachment_index, SetupAttachmentNameIndex)" in helper
    assert "setup_attachment_index.require(" in helper
    assert "attachment_names_by_slot" not in helper
    assert "available_names" not in helper


def test_attachment_time_is_finite_and_non_decreasing():
    helper = slot_attachment_helper_source()

    assert 'time_value = keyframe.get("time", 0)' in helper
    assert "isinstance(time_value, bool)" in helper
    assert "if not _is_finite_number(time_value):" in helper
    assert "time_value < previous_time" in helper
    assert "time_value <= previous_time" not in helper


def test_null_or_omitted_name_hides_attachment_without_fake_default():
    helper = slot_attachment_helper_source()

    assert 'if "name" not in keyframe or keyframe["name"] is None:' in helper
    assert '_require_name(attachment_name, f"{keyframe_path}.name")' in helper
    assert "setup_attachment_index.require(" in helper
    assert "setdefault(" not in helper
    assert 'keyframe["name"] =' not in helper
    assert 'keyframe.get("name", None)' not in helper


def test_slot_attachment_contract_does_not_normalize_or_move_timelines():
    helper = slot_attachment_helper_source()

    assert ".strip()" not in helper
    assert ".lower()" not in helper
    assert ".sort(" not in helper
    assert "sorted(" not in helper
    assert 'keyframe["time"] =' not in helper
    assert 'slot_metadata["attachment"] =' not in helper


def test_grouped_camera_remains_owner_of_hidden_source_slot_timelines():
    source = read(GROUPED_OVERLAY)
    retained_start = source.index("def _retained_slot_mapping(")
    strip_start = source.index("def _strip_hidden_visual_timelines(", retained_start)
    apply_start = source.index("def apply_grouped_camera_overlay(", strip_start)
    retained_source = source[retained_start:strip_start]
    strip_source = source[strip_start:apply_start]
    apply_source = source[apply_start:]

    assert "hidden_slot_names" in retained_source
    assert "if str(slot_name) not in hidden_slot_names" in retained_source
    assert "_retained_slot_mapping(" in strip_source
    assert 'copied_payload.pop("slots", None)' in strip_source
    assert "animations=_strip_hidden_visual_timelines(" in apply_source
    assert "hidden_set" in apply_source


def test_serializer_keeps_slot_attachment_timelines_as_supplied():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    assert 'data["animations"] = dict(document.animations)' in to_dict_source
    assert 'document.animations.setdefault(' not in to_dict_source
