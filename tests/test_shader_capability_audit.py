from Blender_to_Spine2D_Mesh_Exporter.blender_adapter import (
    audit_material_graph_capabilities,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    MaterialDependencyKind,
    MaterialGraphSnapshot,
    MaterialSemanticChannel,
    ShaderBakeCapability,
    ShaderLinkSnapshot,
    ShaderNodeSnapshot,
)


def _graph(
    node_type,
    *,
    output_socket="Value",
    channels=(MaterialSemanticChannel.SURFACE_COLOR,),
    dependencies=(),
    issues=(),
):
    source = ShaderNodeSnapshot(
        node_id="Source",
        node_type=node_type,
        node_name="Source",
    )
    output = ShaderNodeSnapshot(
        node_id="Material Output",
        node_type="OUTPUT_MATERIAL",
        node_name="Material Output",
    )
    return MaterialGraphSnapshot(
        material_name="Material",
        active_output_node_id=output.node_id,
        reachable_nodes=(source, output),
        reachable_links=(
            ShaderLinkSnapshot(
                from_node_id=source.node_id,
                from_socket=output_socket,
                to_node_id=output.node_id,
                to_socket="Surface",
            ),
        ),
        semantic_channels=tuple(channels),
        dependencies=tuple(dependencies),
        issues=tuple(issues),
    )


def _codes(audit):
    return {finding.code for finding in audit.findings}


def test_local_image_graph_is_audited_as_local_uv_safe():
    audit = audit_material_graph_capabilities(
        _graph("TEX_IMAGE", output_socket="Color"),
        render_target="CYCLES",
    )

    assert audit.required_capability is ShaderBakeCapability.LOCAL_UV_SAFE


def test_common_normal_and_vector_nodes_remain_local_safe():
    for node_type in ("NORMAL_MAP", "BUMP", "TANGENT", "VECTOR_ROTATE"):
        audit = audit_material_graph_capabilities(
            _graph(node_type, output_socket="Normal"),
            render_target="CYCLES",
        )
        assert audit.required_capability is ShaderBakeCapability.LOCAL_UV_SAFE, (
            node_type,
            audit,
        )


def test_vector_transform_and_hair_closures_require_camera_render():
    for node_type in (
        "VECT_TRANSFORM",
        "BSDF_HAIR",
        "BSDF_HAIR_PRINCIPLED",
        "BSDF_ANISOTROPIC",
    ):
        audit = audit_material_graph_capabilities(
            _graph(node_type),
            render_target="CYCLES",
        )
        assert audit.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED, (
            node_type,
            audit,
        )


def test_texture_coordinate_uv_is_local_but_window_requires_camera():
    local = audit_material_graph_capabilities(
        _graph("TEX_COORD", output_socket="UV"),
        render_target="CYCLES",
    )
    camera = audit_material_graph_capabilities(
        _graph("TEX_COORD", output_socket="Window"),
        render_target="CYCLES",
    )

    assert local.required_capability is ShaderBakeCapability.LOCAL_UV_SAFE
    assert camera.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
    assert "TEXTURE_COORD_SOURCE_CONTEXT" in _codes(camera)


def test_texture_coordinate_from_instancer_requires_group_render():
    audit = audit_material_graph_capabilities(
        _graph("TEX_COORD", output_socket="From Instancer"),
        render_target="CYCLES",
    )

    assert audit.required_capability is ShaderBakeCapability.GROUP_RENDER_REQUIRED


def test_camera_data_and_object_info_require_source_camera_render():
    camera = audit_material_graph_capabilities(
        _graph("CAMERA", output_socket="View Vector"),
        render_target="CYCLES",
    )
    object_info = audit_material_graph_capabilities(
        _graph("OBJECT_INFO", output_socket="Random"),
        render_target="CYCLES",
    )

    assert camera.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
    assert object_info.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED


def test_shader_to_rgb_requires_eevee_and_rejects_cycles():
    eevee = audit_material_graph_capabilities(
        _graph("SHADER_TO_RGB", output_socket="Color"),
        render_target="BLENDER_EEVEE",
    )
    cycles = audit_material_graph_capabilities(
        _graph("SHADER_TO_RGB", output_socket="Color"),
        render_target="CYCLES",
    )

    assert eevee.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
    assert "EEVEE_SHADER_TO_RGB" in _codes(eevee)
    assert cycles.required_capability is ShaderBakeCapability.UNSUPPORTED
    assert "SHADER_TO_RGB_RENDERER_MISMATCH" in _codes(cycles)


def test_source_attribute_requires_original_render_geometry():
    audit = audit_material_graph_capabilities(
        _graph("ATTRIBUTE", output_socket="Color"),
        render_target="CYCLES",
    )

    assert audit.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
    assert "SOURCE_ATTRIBUTE_NOT_MATERIALIZED" in _codes(audit)


def test_particle_context_requires_group_render():
    audit = audit_material_graph_capabilities(
        _graph("PARTICLE_INFO", output_socket="Age"),
        render_target="CYCLES",
    )

    assert audit.required_capability is ShaderBakeCapability.GROUP_RENDER_REQUIRED


def test_osl_requires_explicit_preflight():
    audit = audit_material_graph_capabilities(
        _graph("SCRIPT", output_socket="Color"),
        render_target="CYCLES",
    )

    assert audit.required_capability is ShaderBakeCapability.UNSUPPORTED
    assert "OSL_PREFLIGHT_REQUIRED" in _codes(audit)


def test_graph_issue_makes_incomplete_analysis_unsupported():
    audit = audit_material_graph_capabilities(
        _graph(
            "TEX_IMAGE",
            output_socket="Color",
            issues=("Unable to map group output",),
        ),
        render_target="CYCLES",
    )

    assert audit.required_capability is ShaderBakeCapability.UNSUPPORTED
    assert "GRAPH_ANALYSIS_INCOMPLETE" in _codes(audit)


def test_volume_and_scene_dependencies_raise_the_required_boundary():
    volume = audit_material_graph_capabilities(
        _graph(
            "PRINCIPLED_VOLUME",
            output_socket="Volume",
            channels=(MaterialSemanticChannel.VOLUME,),
        ),
        render_target="CYCLES",
    )
    scene = audit_material_graph_capabilities(
        _graph(
            "TEX_IMAGE",
            output_socket="Color",
            dependencies=(MaterialDependencyKind.LIGHTING,),
        ),
        render_target="CYCLES",
    )

    assert volume.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
    assert "VOLUME_RENDER_REQUIRED" in _codes(volume)
    assert scene.required_capability is ShaderBakeCapability.SCENE_UV_SAFE


def test_unclassified_reachable_node_is_not_silently_treated_as_local():
    audit = audit_material_graph_capabilities(
        _graph("FUTURE_BLENDER_NODE"),
        render_target="CYCLES",
    )

    assert audit.required_capability is ShaderBakeCapability.UNSUPPORTED
    assert "UNCLASSIFIED_REACHABLE_NODE" in _codes(audit)
