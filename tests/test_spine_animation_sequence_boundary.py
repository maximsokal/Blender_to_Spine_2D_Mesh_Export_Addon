from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "sequence_timeline_contract.py"
)
SERIALIZER = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "serializer.py"
)
MODEL = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "model.py"
)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_sequence_contract_owns_exact_canonical_modes():
    source = read(CONTRACT)

    for mode in (
        "hold",
        "once",
        "loop",
        "pingpong",
        "onceReverse",
        "loopReverse",
        "pingpongReverse",
    ):
        assert f'"{mode}"' in source

    assert 'mode = keyframe.get("mode", "hold")' in source
    assert "if mode not in _SEQUENCE_MODES:" in source
    assert ".lower()" not in source
    assert ".upper()" not in source


def test_sequence_contract_owns_texture_region_attachment_types():
    source = read(CONTRACT)

    assert (
        '_TEXTURE_REGION_ATTACHMENT_TYPES = frozenset('
        '{"region", "mesh", "linkedmesh"}'
        ')' in source
    )
    assert "non-sequence attachment type" in source
    assert 'attachment.get("sequence")' in source
    assert "sequence is required for a sequence timeline" in source


def test_setup_sequence_count_is_fail_closed():
    source = read(CONTRACT)

    assert 'if "count" not in sequence:' in source
    assert "isinstance(count, bool)" in source
    assert "not isinstance(count, int)" in source
    assert "if count < 1:" in source


def test_index_bound_matches_exact_float_frame_packing():
    source = read(CONTRACT)

    assert "_SEQUENCE_INDEX_MAX = ((1 << 24) - 1) >> 4" in source
    assert "mode | (index << 4)" in source
    assert "index > _SEQUENCE_INDEX_MAX" in source
    assert "exact runtime frame packing" in source


def test_delay_inheritance_matches_spine_loader():
    source = read(CONTRACT)

    assert "last_delay: float | int = 0" in source
    assert 'if "delay" in keyframe:' in source
    assert "delay = last_delay" in source
    assert 'if mode != "hold" and delay <= 0:' in source
    assert "last_delay = delay" in source


def test_sequence_scalars_reuse_shared_strict_requirements():
    source = read(CONTRACT)

    assert "from .spine_scalar_contract import (" in source
    assert "require_finite_number as _require_finite_number" in source
    assert "require_name as _require_name" in source
    assert "def _require_finite_number(" not in source
    assert "def _require_name(" not in source
    assert "from math import isfinite" not in source
    assert 'keyframe.get("time", 0)' in source
    assert "time_value < previous_time" in source
    assert "time_value <= previous_time" not in source
    assert "_require_name(skin_name" in source
    assert "_require_name(" in source


def test_reference_chain_uses_shared_setup_indices():
    source = read(CONTRACT)

    assert "AttachmentReference" in source
    assert "LinkedMeshResolver" in source
    assert "SetupSlotIndex" in source
    assert "resolve_setup_slot_index" in source
    assert "resolver.require_skin(skin_name, path=skin_path)" in source
    assert "slot_index.require(slot_name, path=slot_path)" in source
    assert "reference = AttachmentReference(" in source
    assert "setup = resolver.get_attachment(" in source
    assert "setup.attachment" in source
    assert "def _build_skin_index(" not in source
    assert "def _resolve_setup_attachment(" not in source
    assert "known_slot_names" not in source
    assert "ambiguous_slot_names" not in source


def test_serializer_runs_sequence_contract_after_existing_boundaries():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    validator_index = to_dict_source.index(
        "self._validator.validate_or_raise(document)"
    )
    color_index = to_dict_source.index(
        "validate_animation_slot_color_timelines("
    )
    curve_index = to_dict_source.index("validate_animation_curves(")
    deform_index = to_dict_source.index(
        "validate_animation_deform_timelines("
    )
    sequence_index = to_dict_source.index(
        "validate_animation_sequence_timelines("
    )
    data_index = to_dict_source.index('data: dict[str, Any] = {')

    assert (
        validator_index
        < color_index
        < curve_index
        < deform_index
        < sequence_index
        < data_index
    )
    assert "skins=document.skins" in to_dict_source
    assert "slot_names=slot_names" in to_dict_source
    sequence_call = to_dict_source[sequence_index:data_index]
    assert "linked_mesh_resolver=linked_mesh_resolver" in sequence_call
    assert "setup_slot_index=setup_slot_index" in sequence_call


def test_sequence_contract_is_output_boundary_not_model_mutation():
    model_source = read(MODEL)
    serializer_source = read(SERIALIZER)

    assert "sequence_timeline_contract" not in model_source
    assert (
        "from .sequence_timeline_contract import "
        "validate_animation_sequence_timelines"
    ) in serializer_source


def test_sequence_contract_never_normalizes_payloads():
    source = read(CONTRACT)

    for forbidden in (
        ".sort(",
        "setdefault(",
        'keyframe["time"] =',
        'keyframe["mode"] =',
        'keyframe["index"] =',
        'keyframe["delay"] =',
        ".lower()",
        ".upper()",
    ):
        assert forbidden not in source


def test_unknown_attachment_timelines_remain_outside_contract():
    source = read(CONTRACT)

    assert 'if "sequence" not in attachment_metadata:' in source
    assert 'attachment_metadata["sequence"]' in source
    assert 'attachment_metadata["deform"]' not in source
    assert "unknown attachment timelines" in source


def test_serializer_keeps_animation_mapping_as_supplied():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    assert 'data["animations"] = dict(document.animations)' in to_dict_source
    assert 'document.animations.setdefault(' not in to_dict_source
    assert 'document.animations["attachments"] =' not in to_dict_source
