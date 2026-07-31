"""Animation mix translation for the Spine 3.8 codec."""

import json

from Blender_to_Spine2D_Mesh_Exporter.domain.spine.model import (
    Bone,
    Skin,
    SpineDocument,
    TransformConstraint,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_codecs import (
    serialize_spine_document,
)
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.version_target import SpineJsonTarget


def test_spine38_rewrites_transform_animation_frame_mixes() -> None:
    document = SpineDocument(
        skeleton={"spine": "4.2.43"},
        bones=(Bone("root"), Bone("control", parent="root")),
        slots=(),
        skins=(Skin("default", {}),),
        transform=(
            TransformConstraint("copy", 0, ("control",), "root"),
        ),
        animations={
            "animation": {
                "transform": {
                    "copy": [
                        {
                            "time": 0.0,
                            "mixRotate": 0.1,
                            "mixX": 0.2,
                            "mixY": 0.2,
                            "mixScaleX": 0.3,
                            "mixScaleY": 0.3,
                            "mixShearY": 0.4,
                            "curve": "stepped",
                        }
                    ]
                }
            }
        },
    )

    payload = json.loads(
        serialize_spine_document(document, SpineJsonTarget.SPINE_3_8)
    )
    frame = payload["animations"]["animation"]["transform"]["copy"][0]

    assert frame == {
        "time": 0.0,
        "curve": "stepped",
        "rotateMix": 0.1,
        "translateMix": 0.2,
        "scaleMix": 0.3,
        "shearMix": 0.4,
    }
