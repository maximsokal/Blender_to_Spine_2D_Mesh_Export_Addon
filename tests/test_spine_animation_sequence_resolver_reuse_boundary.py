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


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_sequence_contract_uses_shared_setup_attachment_index():
    source = read(CONTRACT)

    assert "AttachmentReference" in source
    assert "LinkedMeshResolver" in source
    assert "raw_attachment_type" in source
    assert "resolver.require_skin(skin_name, path=skin_path)" in source
    assert "setup = resolver.get_attachment(" in source
    assert "setup.attachment" in source

    assert "def _build_skin_index(" not in source
    assert "def _resolve_setup_attachment(" not in source
    assert "skin_by_name" not in source
    assert "ambiguous_skin_names" not in source


def test_sequence_contract_accepts_reusable_resolver_without_owning_setup_validation():
    source = read(CONTRACT)

    assert (
        "linked_mesh_resolver: LinkedMeshResolver | None = None"
        in source
    )
    assert (
        'resolver = LinkedMeshResolver(skins, path="document.skins")'
        in source
    )
    assert (
        "linked_mesh_resolver must be LinkedMeshResolver or None"
        in source
    )
    assert "linked_mesh_resolver.skins is not skins" in source
    assert "built from the exact skins tuple" in source
    assert "resolver.validate_all()" not in source


def test_serializer_passes_one_validated_resolver_to_deform_and_sequence():
    source = read(SERIALIZER)
    to_dict_source = source[source.index("def to_dict(") :]

    assert to_dict_source.count("validate_setup_linked_meshes(") == 1
    assert (
        "linked_mesh_resolver = validate_setup_linked_meshes("
        in to_dict_source
    )

    deform_call = to_dict_source[
        to_dict_source.index("validate_animation_deform_timelines(") :
        to_dict_source.index("validate_animation_sequence_timelines(")
    ]
    sequence_call = to_dict_source[
        to_dict_source.index("validate_animation_sequence_timelines(") :
        to_dict_source.index('data: dict[str, Any] = {')
    ]

    assert "linked_mesh_resolver=linked_mesh_resolver" in deform_call
    assert "linked_mesh_resolver=linked_mesh_resolver" in sequence_call


def test_sequence_specific_semantics_remain_local_to_sequence_contract():
    source = read(CONTRACT)

    assert "_SEQUENCE_MODES" in source
    assert "_SEQUENCE_INDEX_MAX" in source
    assert "_TEXTURE_REGION_ATTACHMENT_TYPES" in source
    assert "def _resolve_setup_sequence(" in source
    assert "def _validate_sequence_timeline(" in source
    assert 'mode = keyframe.get("mode", "hold")' in source
    assert 'index = keyframe.get("index", 0)' in source
    assert 'if "delay" in keyframe:' in source
    assert "delay = last_delay" in source
