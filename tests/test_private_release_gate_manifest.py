import pytest

from Blender_to_Spine2D_Mesh_Exporter.domain.release_gate import (
    PrivateReleaseGateError,
    parse_private_release_manifest,
)


def _fixture(fixture_id="fixture-a", capabilities=None):
    return {
        "id": fixture_id,
        "source_blend": f"{fixture_id}/source.blend",
        "legacy_json": f"{fixture_id}/legacy.json",
        "actual_json": f"{fixture_id}_merged.json",
        "operator": "object.save_uv_as_json",
        "active_object": "Character",
        "selected_objects": ["Character"],
        "capabilities": capabilities or ["single", "b4_cycles"],
        "image_pairs": [
            {
                "expected": f"{fixture_id}/legacy-images/Character_Baked.png",
                "actual": "images/Character_Baked.png",
                "maximum_absolute_error": 0.02,
                "mean_absolute_error": 0.005,
                "alpha_maximum_absolute_error": 0.01,
            }
        ],
        "scene_attributes": {"render.engine": "CYCLES"},
        "scene_custom_properties": {"spine2d_output_dir": "${OUTPUT_DIR}"},
        "object_attributes": {},
        "object_custom_properties": {},
        "operator_kwargs": {},
        "ignored_paths": [],
        "accepted_warning_codes": [],
        "strict_edges": True,
        "compare_animations": False,
        "animated": False,
    }


def _manifest(fixtures=None, required=None, minimum=1):
    return {
        "schema_version": 1,
        "suite_id": "private-production-v1",
        "blender_version": "4.4.0",
        "release_gate": {
            "minimum_fixture_count": minimum,
            "required_capabilities": required or ["single", "b4_cycles"],
            "require_strict_edges": True,
            "require_animation_parity_for_animated": True,
            "allow_unaccepted_warnings": False,
        },
        "fixtures": fixtures or [_fixture()],
    }


def test_valid_manifest_resolves_strict_private_gate_contract():
    manifest = parse_private_release_manifest(_manifest())

    assert manifest.schema_version == 1
    assert manifest.suite_id == "private-production-v1"
    assert manifest.blender_version == "4.4.0"
    assert len(manifest.fixtures) == 1
    fixture = manifest.fixtures[0]
    assert fixture.fixture_id == "fixture-a"
    assert fixture.strict_edges
    assert fixture.image_pairs[0].maximum_absolute_error == 0.02


def test_manifest_rejects_fixture_path_escape():
    value = _manifest()
    value["fixtures"][0]["source_blend"] = "../private/source.blend"

    with pytest.raises(PrivateReleaseGateError, match="cannot escape"):
        parse_private_release_manifest(value)


def test_manifest_rejects_missing_required_capability():
    with pytest.raises(PrivateReleaseGateError, match="missing required capabilities"):
        parse_private_release_manifest(
            _manifest(required=["single", "grouped_b4"])
        )


def test_manifest_rejects_too_few_private_fixtures():
    with pytest.raises(PrivateReleaseGateError, match="fixture count"):
        parse_private_release_manifest(_manifest(minimum=2))


def test_release_gate_requires_strict_edges_for_every_fixture():
    value = _manifest()
    value["fixtures"][0]["strict_edges"] = False

    with pytest.raises(PrivateReleaseGateError, match="strict edges"):
        parse_private_release_manifest(value)


def test_animated_fixture_must_compare_animations():
    value = _manifest()
    value["fixtures"][0]["animated"] = True
    value["fixtures"][0]["compare_animations"] = False

    with pytest.raises(PrivateReleaseGateError, match="compare animations"):
        parse_private_release_manifest(value)


def test_active_object_must_be_selected():
    value = _manifest()
    value["fixtures"][0]["selected_objects"] = ["Other"]

    with pytest.raises(PrivateReleaseGateError, match="also appear"):
        parse_private_release_manifest(value)


def test_fixture_ids_must_be_unique():
    first = _fixture("same")
    second = _fixture("same")

    with pytest.raises(PrivateReleaseGateError, match="fixture ids"):
        parse_private_release_manifest(
            _manifest(fixtures=[first, second], minimum=2)
        )


def test_operator_must_use_module_operator_shape():
    value = _manifest()
    value["fixtures"][0]["operator"] = "save_uv_as_json"

    with pytest.raises(PrivateReleaseGateError, match="module.operator"):
        parse_private_release_manifest(value)


def test_image_tolerances_must_be_finite_and_non_negative():
    value = _manifest()
    value["fixtures"][0]["image_pairs"][0]["mean_absolute_error"] = -0.1

    with pytest.raises(PrivateReleaseGateError, match="finite and non-negative"):
        parse_private_release_manifest(value)
