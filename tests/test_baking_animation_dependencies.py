from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    MaterialAnalysis,
    MaterialDependencyKind,
    MaterialGraphSnapshot,
    MaterialKind,
    MaterialSemanticChannel,
    ObjectMaterialAnalysis,
    ShaderNodeSnapshot,
)


def test_time_dependency_marks_material_and_object_as_animated():
    output = ShaderNodeSnapshot(
        node_id="Material Output",
        node_type="OUTPUT_MATERIAL",
        node_name="Material Output",
    )
    graph = MaterialGraphSnapshot(
        material_name="Animated Material",
        active_output_node_id=output.node_id,
        reachable_nodes=(output,),
        reachable_links=(),
        semantic_channels=(MaterialSemanticChannel.SURFACE_COLOR,),
        dependencies=(MaterialDependencyKind.TIME,),
    )
    material = MaterialAnalysis(
        slot_index=0,
        material_name="Animated Material",
        kind=MaterialKind.SOLID_COLOR,
        graph=graph,
    )
    analysis = ObjectMaterialAnalysis("AnimatedObject", (material,))

    assert material.animated
    assert analysis.has_animated_dependencies
