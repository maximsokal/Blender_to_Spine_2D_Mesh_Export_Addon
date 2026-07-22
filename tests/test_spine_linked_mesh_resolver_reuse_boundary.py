from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINKED = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "linked_mesh_contract.py"
)
DEFORM = (
    ROOT
    / "Blender_to_Spine2D_Mesh_Exporter"
    / "domain"
    / "spine"
    / "deform_timeline_contract.py"
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


def test_resolver_retains_the_exact_skin_tuple_used_for_indexing():
    source = read(LINKED)

    assert "self._skins = skins" in source
    assert "def skins(self) -> tuple[Skin, ...]:" in source
    assert "return self._skins" in source
    assert "tuple(skins)" not in source
    assert "list(skins)" not in source


def test_setup_validation_returns_the_already_validated_resolver():
    source = read(LINKED)
    function_start = source.index("def validate_setup_linked_meshes(")
    function_source = source[function_start:]

    assert ") -> LinkedMeshResolver:" in function_source
    assert "resolver = LinkedMeshResolver(skins, path=path)" in function_source
    assert "resolver.validate_all()" in function_source
    assert "return resolver" in function_source
    assert function_source.index("resolver.validate_all()") < function_source.index(
        "return resolver"
    )


def test_deform_boundary_accepts_optional_shared_resolver():
    source = read(DEFORM)

    assert "linked_mesh_resolver: LinkedMeshResolver | None = None" in source
    assert "if linked_mesh_resolver is None:" in source
    assert 'resolver = LinkedMeshResolver(skins, path="document.skins")' in source
    assert "isinstance(linked_mesh_resolver, LinkedMeshResolver)" in source
    assert "if linked_mesh_resolver.skins is not skins:" in source
    assert "must be built from the exact skins tuple" in source
    assert "resolver = linked_mesh_resolver" in source


def test_serializer_passes_one_setup_resolver_to_deform_boundary():
    source = read(SERIALIZER)
    to_dict_start = source.index("def to_dict(")
    to_dict_source = source[to_dict_start:]

    assignment = "linked_mesh_resolver = validate_setup_linked_meshes("
    deform_call = "validate_animation_deform_timelines("
    resolver_argument = "linked_mesh_resolver=linked_mesh_resolver"

    assert assignment in to_dict_source
    assert deform_call in to_dict_source
    assert resolver_argument in to_dict_source
    assert to_dict_source.count("validate_setup_linked_meshes(") == 1
    assert to_dict_source.index(assignment) < to_dict_source.index(deform_call)
    assert to_dict_source.index(deform_call) < to_dict_source.index(
        resolver_argument
    )


def test_deform_capacity_cache_uses_terminal_and_source_references():
    source = read(DEFORM)

    assert "capacity_reference = reference" in source
    assert "capacity_reference = resolved.terminal" in source
    assert "terminal_cached = cache.get(capacity_reference)" in source
    assert "cache[reference] = terminal_cached" in source
    assert "cache[capacity_reference] = capacity" in source
    assert "cache[reference] = capacity" in source


def test_deform_contract_does_not_build_a_second_resolver_when_one_is_supplied():
    source = read(DEFORM)
    resolver_branch_start = source.index("if linked_mesh_resolver is None:")
    slot_index_start = source.index(
        "slot_index = resolve_setup_slot_index(",
        resolver_branch_start,
    )
    resolver_branch = source[resolver_branch_start:slot_index_start]

    assert resolver_branch.count("LinkedMeshResolver(") == 1
    assert "resolver = linked_mesh_resolver" in resolver_branch
    assert "LinkedMeshResolver(linked_mesh_resolver" not in resolver_branch
