from dataclasses import replace

from Blender_to_Spine2D_Mesh_Exporter.domain.spine import (
    LegacyAttachmentVertex,
    LegacyMeshAttachmentRequest,
    LegacyRigBuildRequest,
    LegacyZGroup,
    SpineSerializer,
    SpineValidator,
    build_legacy_mesh_document,
    build_legacy_rig,
    decode_weighted_vertices,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.vertex_bone_optimizer import (
    optimize_shared_vertex_bones,
)


def _rig():
    return build_legacy_rig(
        LegacyRigBuildRequest(
            prefix="Cone",
            texture_width=128,
            texture_height=128,
            z_groups=(
                LegacyZGroup(0.0, height_real_pixels=0.0),
                LegacyZGroup(1.0, height_real_pixels=256.0),
            ),
        )
    )


_POINTS = {
    "A": ((0.0, -128.0), 1),
    "B": ((110.85, 64.0), 1),
    "C": ((-110.85, 64.0), 1),
    "D": ((0.0, 0.0), 2),
}


def _request(segment_index: int, point_names: tuple[str, str, str]):
    name = f"Cone_Segment_{segment_index}"
    vertices = tuple(
        LegacyAttachmentVertex(
            index,
            (
                float((segment_index + index) % 3) / 2.0,
                float(index) / 2.0,
            ),
            _POINTS[point_name][0],
            _POINTS[point_name][1],
        )
        for index, point_name in enumerate(point_names)
    )
    return LegacyMeshAttachmentRequest(
        slot_name=name,
        attachment_name=name,
        vertex_prefix=name,
        image_path="images/Cone_Baked",
        width=128.0,
        height=128.0,
        vertices=vertices,
        triangles=(0, 1, 2),
        hull=3,
        edges=(0, 1, 1, 2, 2, 0),
    )


def _pyramid_requests():
    return (
        _request(0, ("A", "B", "D")),
        _request(1, ("A", "B", "C")),
        _request(2, ("B", "C", "D")),
        _request(3, ("A", "C", "D")),
    )


def _indices(component):
    return tuple(
        vertex.influences[0].bone_index
        for vertex in decode_weighted_vertices(
            component.attachment.vertices,
            expected_vertex_count=3,
        )
    )


def test_pyramid_twelve_segment_bones_compact_to_four_shared_bones():
    rig = _rig()
    original = build_legacy_mesh_document(rig, _pyramid_requests())
    base = len(rig.bones)

    assert len(original.document.bones) == base + 12

    optimized = optimize_shared_vertex_bones(original)

    assert len(optimized.document.bones) == base + 4
    assert tuple(bone.name for bone in optimized.document.bones[base:]) == (
        "Cone_Segment_0_vertex_0",
        "Cone_Segment_0_vertex_1",
        "Cone_Segment_0_vertex_2",
        "Cone_Segment_1_vertex_2",
    )
    assert tuple(_indices(component) for component in optimized.components) == (
        (base, base + 1, base + 2),
        (base, base + 1, base + 3),
        (base + 1, base + 3, base + 2),
        (base, base + 3, base + 2),
    )
    assert SpineValidator().validate(optimized.document) == ()


def test_weight_remap_preserves_mesh_geometry_uvs_and_local_influence_data():
    original = build_legacy_mesh_document(_rig(), _pyramid_requests())
    optimized = optimize_shared_vertex_bones(original)

    for before, after in zip(original.components, optimized.components, strict=True):
        assert after.attachment.uvs == before.attachment.uvs
        assert after.attachment.triangles == before.attachment.triangles
        assert after.attachment.hull == before.attachment.hull
        assert after.attachment.edges == before.attachment.edges
        assert after.attachment.path == before.attachment.path
        assert after.attachment.width == before.attachment.width
        assert after.attachment.height == before.attachment.height

        decoded = decode_weighted_vertices(
            after.attachment.vertices,
            expected_vertex_count=len(after.request.vertices),
        )
        assert all(
            len(vertex.influences) == 1
            and (
                vertex.influences[0].x,
                vertex.influences[0].y,
                vertex.influences[0].weight,
            )
            == (0.0, 0.0, 1.0)
            for vertex in decoded
        )

    serialized = SpineSerializer().to_dict(optimized.document)
    attachments = serialized["skins"][0]["attachments"]
    assert tuple(attachments) == tuple(
        f"Cone_Segment_{index}" for index in range(4)
    )
    assert all(
        attachments[name][name]["triangles"] == [0, 1, 2]
        for name in attachments
    )


def test_same_xy_in_different_z_parents_remains_independent():
    rig = _rig()
    first = _request(0, ("A", "B", "C"))
    second_base = _request(1, ("A", "B", "C"))
    second = replace(
        second_base,
        vertices=tuple(
            replace(vertex, z_group_index=2)
            for vertex in second_base.vertices
        ),
    )
    original = build_legacy_mesh_document(rig, (first, second))

    optimized = optimize_shared_vertex_bones(original)

    assert optimized is original
    assert len(optimized.document.bones) == len(rig.bones) + 6


def test_optimizer_is_idempotent_after_first_compaction():
    optimized = optimize_shared_vertex_bones(
        build_legacy_mesh_document(_rig(), _pyramid_requests())
    )

    assert optimize_shared_vertex_bones(optimized) is optimized
