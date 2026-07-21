from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "deform_timeline_contract.py"
)
CURVES = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "curve_timeline_contract.py"
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


def test_deform_contract_owns_exact_vertex_attachment_types():
    source = read(CONTRACT)

    assert (
        '_VERTEX_ATTACHMENT_TYPES = frozenset('
        '{"mesh", "linkedmesh", "boundingbox", "path", "clipping"}'
        ')' in source
    )
    assert (
        '_VERTEX_COUNT_ATTACHMENT_TYPES = frozenset('
        '{"boundingbox", "path", "clipping"}'
        ')' in source
    )
    assert 'attachment.get("type", "region")' in source
    assert "non-deformable attachment type" in source


def test_unweighted_and_weighted_capacity_have_distinct_ownership():
    source = read(CONTRACT)

    assert "if len(stream) == expected_coordinate_count:" in source
    assert "return expected_coordinate_count" in source
    assert "decode_weighted_vertices(" in source
    assert "expected_vertex_count=expected_coordinate_count // 2" in source
    assert "sum(len(vertex.influences) for vertex in decoded) * 2" in source


def test_mesh_and_vertex_count_attachments_resolve_setup_coordinate_count():
    source = read(CONTRACT)

    assert 'if attachment_type in {"mesh", "linkedmesh"}:' in source
    assert 'if "uvs" not in attachment:' in source
    assert "return len(uvs)" in source
    assert "if attachment_type in _VERTEX_COUNT_ATTACHMENT_TYPES:" in source
    assert 'if "vertexCount" not in attachment:' in source
    assert "return vertex_count * 2" in source


def test_linked_mesh_resolution_matches_runtime_default_skin_policy():
    source = read(CONTRACT)

    assert 'attachment.get("parent")' in source
    assert 'attachment.get("skin")' in source
    assert 'if raw_parent_skin in (None, ""):' in source
    assert 'parent_skin_name = "default"' in source
    assert "linked mesh parent cycle" in source
    assert "capacity_cache" in source
    assert "resolving=set()" in source


def test_animation_reference_chain_is_fail_closed():
    source = read(CONTRACT)

    assert 'if "attachments" not in animation_metadata:' in source
    assert "skin_name not in skin_by_name" in source
    assert "slot_name not in known_slot_names" in source
    assert "slot_attachments.get(attachment_name)" in source
    assert "references undefined attachment" in source
    assert "references duplicated skin" in source
    assert "references duplicated setup slot" in source


def test_consumed_vertices_and_offset_preserve_xy_pairs_and_capacity():
    source = read(CONTRACT)

    assert 'if "vertices" in keyframe:' in source
    assert "if len(deform_vertices) % 2:" in source
    assert "vertices must contain" in source
    assert "_require_non_negative_even_int(" in source
    assert "offset + len(deform_vertices)" in source
    assert "if end > capacity:" in source
    assert "exceeds deform capacity" in source


def test_offset_is_not_read_without_vertices():
    source = read(CONTRACT)
    keyframe_start = source.index('if "vertices" in keyframe:')
    curve_start = source.index(
        "# Spine consumes a curve only when a next keyframe exists."
    )
    consumed_block = source[keyframe_start:curve_start]

    assert 'keyframe.get("offset", 0)' in consumed_block
    assert 'keyframe.get("offset", 0)' not in source[:keyframe_start]
    assert 'keyframe["offset"]' not in source


def test_deform_time_and_components_are_strict_finite_numbers():
    source = read(CONTRACT)

    assert "isinstance(value, bool)" in source
    assert "not isinstance(value, (int, float))" in source
    assert "not isfinite(value)" in source
    assert 'time_value = keyframe.get("time", 0)' in source
    assert "time_value < previous_time" in source
    assert "time_value <= previous_time" not in source


def test_deform_curve_reuses_shared_single_channel_contract():
    source = read(CONTRACT)
    curve_source = read(CURVES)

    assert "from .curve_timeline_contract import validate_curve_value" in source
    assert "validate_curve_value(" in source
    assert "channel_count=1" in source
    assert "keyframe_index < last_keyframe_index" in source
    assert 'def validate_curve_value(' in curve_source
    assert '__all__ = ["validate_animation_curves", "validate_curve_value"]' in (
        curve_source
    )


def test_serializer_runs_deform_contract_after_existing_boundaries():
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
    data_index = to_dict_source.index('data: dict[str, Any] = {')

    assert validator_index < color_index < curve_index < deform_index < data_index
    assert "skins=document.skins" in to_dict_source
    assert "slot_names=tuple(slot.name for slot in document.slots)" in (
        to_dict_source
    )
    assert 'path="document.animations"' in to_dict_source


def test_deform_contract_is_output_boundary_not_model_mutation():
    model_source = read(MODEL)
    serializer_source = read(SERIALIZER)

    assert "deform_timeline_contract" not in model_source
    assert (
        "from .deform_timeline_contract import "
        "validate_animation_deform_timelines"
    ) in serializer_source


def test_deform_contract_never_normalizes_payloads():
    source = read(CONTRACT)

    for forbidden in (
        ".sort(",
        "sorted(",
        "setdefault(",
        'keyframe["vertices"] =',
        'keyframe["offset"] =',
        'keyframe["time"] =',
        'keyframe["curve"] =',
        ".lower()",
        ".upper()",
    ):
        assert forbidden not in source


def test_unknown_attachment_timelines_remain_outside_contract():
    source = read(CONTRACT)

    assert 'if "deform" not in attachment_metadata:' in source
    assert 'attachment_metadata["deform"]' in source
    assert 'attachment_metadata["sequence"]' not in source
    assert "Unknown attachment" in source


def test_serializer_keeps_animation_mapping_as_supplied():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    assert 'data["animations"] = dict(document.animations)' in to_dict_source
    assert 'document.animations.setdefault(' not in to_dict_source
    assert 'document.animations["attachments"] =' not in to_dict_source
