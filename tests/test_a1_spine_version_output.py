"""Common target contracts for multi- and mixed-object A1 output."""

from __future__ import annotations

from pathlib import Path

import pytest

from Blender_to_Spine2D_Mesh_Exporter.application import (
    A1SingleObjectExportSettings,
    ExportSettings,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_multi_object_contracts import (
    A1MultiObjectSource,
)
from Blender_to_Spine2D_Mesh_Exporter.blender_adapter.a1_spine_version_output import (
    resolve_a1_sources_spine_target,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def _source(
    root: Path,
    component_id: str,
    target: SpineJsonTarget,
) -> A1MultiObjectSource:
    return A1MultiObjectSource(
        source_object=object(),
        component_id=component_id,
        animation_namespace=component_id,
        settings=A1SingleObjectExportSettings(
            export=ExportSettings(
                texture_width=64,
                texture_height=64,
                output_directory=root,
                spine_version=target.exact_version,
            )
        ),
    )


@pytest.mark.parametrize("target", tuple(SpineJsonTarget))
def test_registered_targets_can_cross_the_immutable_request_boundary(
    tmp_path: Path,
    target: SpineJsonTarget,
) -> None:
    source = _source(tmp_path, "component", target)

    assert source.settings.export.spine_target is target
    assert resolve_a1_sources_spine_target((source,)) is target


def test_multi_sources_must_use_one_target(tmp_path: Path) -> None:
    sources = (
        _source(tmp_path, "left", SpineJsonTarget.SPINE_4_2),
        _source(tmp_path, "right", SpineJsonTarget.SPINE_4_2),
    )

    assert resolve_a1_sources_spine_target(sources) is SpineJsonTarget.SPINE_4_2


def test_mixed_source_groups_share_the_same_target(tmp_path: Path) -> None:
    connected = (_source(tmp_path, "connected", SpineJsonTarget.SPINE_4_1),)
    standalone = (_source(tmp_path, "standalone", SpineJsonTarget.SPINE_4_1),)

    assert (
        resolve_a1_sources_spine_target(connected, standalone)
        is SpineJsonTarget.SPINE_4_1
    )


def test_target_mismatch_is_rejected_before_output_staging(tmp_path: Path) -> None:
    connected = (_source(tmp_path, "connected", SpineJsonTarget.SPINE_4_2),)
    standalone = (_source(tmp_path, "standalone", SpineJsonTarget.SPINE_3_8),)

    with pytest.raises(ValueError, match="same Spine JSON target") as exc_info:
        resolve_a1_sources_spine_target(connected, standalone)

    message = str(exc_info.value)
    assert "connected=4.2.43" in message
    assert "standalone=3.8.99" in message


@pytest.mark.parametrize("groups", ((), ((),), ([],)))
def test_target_resolver_rejects_missing_or_invalid_source_groups(groups) -> None:
    with pytest.raises((TypeError, ValueError)):
        resolve_a1_sources_spine_target(*groups)
