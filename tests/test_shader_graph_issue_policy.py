"""Regressions for advisory versus blocking shader traversal diagnostics."""

from __future__ import annotations

import pytest

from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.shader_capability_analysis import (
    audit_material_graph_capabilities,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.shader_graph_issue_policy import (
    ShaderGraphIssueSeverity,
    classify_shader_graph_issue,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.baking import (
    MaterialDependencyKind,
    MaterialGraphSnapshot,
    MaterialSemanticChannel,
    ShaderBakeCapability,
    ShaderNodeSnapshot,
)


_MUTED_ADVISORY = (
    "Muted node 'Add Shader' has no unambiguous internal bypass for output "
    "'Shader'; all inputs were analyzed conservatively"
)


def _graph_with_issue(
    issue: str,
    *,
    dependencies: tuple[MaterialDependencyKind, ...] = (),
) -> MaterialGraphSnapshot:
    output = ShaderNodeSnapshot(
        node_id="Material Output",
        node_type="OUTPUT_MATERIAL",
        node_name="Material Output",
    )
    return MaterialGraphSnapshot(
        material_name="Gold coin",
        active_output_node_id=output.node_id,
        reachable_nodes=(output,),
        reachable_links=(),
        semantic_channels=(MaterialSemanticChannel.SURFACE_COLOR,),
        dependencies=dependencies,
        issues=(issue,),
    )


def test_muted_all_inputs_fallback_is_advisory() -> None:
    classification = classify_shader_graph_issue(_MUTED_ADVISORY)

    assert classification.severity is ShaderGraphIssueSeverity.ADVISORY
    assert classification.capability_code == "MUTED_NODE_CONSERVATIVE_ANALYSIS"
    assert not classification.blocks_export


def test_conservative_muted_advisory_does_not_mask_camera_capability() -> None:
    audit = audit_material_graph_capabilities(
        _graph_with_issue(
            _MUTED_ADVISORY,
            dependencies=(MaterialDependencyKind.CAMERA,),
        ),
        render_target="CYCLES",
    )

    assert audit.required_capability is ShaderBakeCapability.CAMERA_RENDER_REQUIRED
    assert "GRAPH_CAMERA_DEPENDENCY" in {
        finding.code for finding in audit.findings
    }
    assert "GRAPH_ANALYSIS_INCOMPLETE" not in {
        finding.code for finding in audit.findings
    }
    assert "MUTED_NODE_CONSERVATIVE_ANALYSIS" not in {
        finding.code for finding in audit.findings
    }


@pytest.mark.parametrize(
    "issue",
    (
        "Reachable node group 'Broken' has no node tree",
        "Node group expansion exceeded 64 levels at 'DeepGroup'",
        "Recursive node group cycle detected: A -> B -> A",
        "Reachable node group 'Broken' has no Group Output node",
        "Unable to map output 'Shader' of node group 'Broken' to its Group Output interface",
        "Unable to map Group Input output 'Color' for node group 'Broken'",
    ),
)
def test_incomplete_group_traversal_remains_blocking(issue: str) -> None:
    classification = classify_shader_graph_issue(issue)
    audit = audit_material_graph_capabilities(
        _graph_with_issue(issue),
        render_target="CYCLES",
    )

    assert classification.severity is ShaderGraphIssueSeverity.BLOCKING
    assert classification.blocks_export
    assert audit.required_capability is ShaderBakeCapability.UNSUPPORTED
    assert tuple(
        (finding.code, finding.reason)
        for finding in audit.findings
        if finding.capability is ShaderBakeCapability.UNSUPPORTED
    ) == (("GRAPH_ANALYSIS_INCOMPLETE", issue),)


def test_issue_policy_rejects_invalid_values() -> None:
    with pytest.raises(TypeError, match="issue must be str"):
        classify_shader_graph_issue(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="issue must be a non-empty string"):
        classify_shader_graph_issue("   ")
