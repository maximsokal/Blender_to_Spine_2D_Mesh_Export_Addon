from __future__ import annotations
from dataclasses import replace
import json, math, os
from random import Random
import pytest
from Blender_to_Spine2D_Mesh_Exporter.domain.geometry import EdgeId,FaceId,LoopId,LoopUV,MeshEdge,MeshFace,MeshLoop,MeshSnapshot,MeshSnapshotValidator,MeshValidationSeverity,MeshVertex,SourceEdgeId,SourceFaceId,SourceLoopId,SourceVertexId,VertexId,build_mesh_fingerprint
from Blender_to_Spine2D_Mesh_Exporter.domain.spine.spine_json_contract import SpineJsonContractError,validate_json_value
DEFAULT_SEED=0x5A17E2D; SEED=int(os.environ.get("SPINE2D_FUZZ_SEED",str(DEFAULT_SEED)),0)
def _ngon(rng:Random,count:int,case_index:int)->MeshSnapshot:
    object_id=f"fuzz-{case_index}"; radius=rng.uniform(0.1,100.0)
    vertices=tuple(MeshVertex(id=VertexId(index),source_id=SourceVertexId(object_id,index),position=(radius*math.cos((2.0*math.pi*index)/count),radius*math.sin((2.0*math.pi*index)/count),rng.uniform(-0.01,0.01)),normal=(0.0,0.0,1.0)) for index in range(count))
    edges=tuple(MeshEdge(id=EdgeId(index),source_id=SourceEdgeId(object_id,index),vertex_ids=(VertexId(index),VertexId((index+1)%count)),seam=bool(rng.getrandbits(1)),sharp=bool(rng.getrandbits(1))) for index in range(count))
    loops=tuple(MeshLoop(id=LoopId(index),source_id=SourceLoopId(object_id,0,index),vertex_id=VertexId(index),edge_id=EdgeId(index),uvs=(LoopUV("UVMap",(rng.uniform(-2.0,3.0),rng.uniform(-2.0,3.0))),)) for index in range(count))
    face=MeshFace(id=FaceId(0),source_id=SourceFaceId(object_id,0),loop_ids=tuple(LoopId(index) for index in range(count)),material_index=rng.randrange(0,4),normal=(0.0,0.0,1.0),smooth=bool(rng.getrandbits(1)))
    return MeshSnapshot(snapshot_id=f"snapshot-{case_index}",source_object_id=object_id,object_name=f"Объект_日本語_{case_index}",vertices=vertices,edges=edges,loops=loops,faces=(face,),uv_layer_names=("UVMap",),active_uv_layer="UVMap",render_uv_layer="UVMap")
def _error_codes(snapshot): return {issue.code for issue in MeshSnapshotValidator().validate(snapshot) if issue.severity is MeshValidationSeverity.ERROR}
def test_seeded_valid_ngons_remain_valid_and_fingerprint_deterministic():
    rng=Random(SEED)
    for case_index in range(200):
        snapshot=_ngon(rng,rng.randint(3,64),case_index); assert not _error_codes(snapshot),f"seed={SEED}, case={case_index}"; assert build_mesh_fingerprint(snapshot)==build_mesh_fingerprint(replace(snapshot)),f"seed={SEED}, case={case_index}"
def test_seeded_topology_corruptions_are_never_silently_accepted():
    rng=Random(SEED^0xBAD5EED); expected_codes={"duplicate":"DUPLICATE_VERTEX_ID","non_dense":"NON_DENSE_EDGE_IDS","missing_vertex":"MISSING_LOOP_VERTEX","missing_edge":"MISSING_LOOP_EDGE"}
    for case_index in range(120):
        snapshot=_ngon(rng,rng.randint(3,24),case_index); mutation=rng.choice(tuple(expected_codes))
        if mutation=="duplicate":
            vertices=list(snapshot.vertices); vertices[-1]=replace(vertices[-1],id=vertices[0].id); corrupted=replace(snapshot,vertices=tuple(vertices))
        elif mutation=="non_dense":
            edges=list(snapshot.edges); edges[-1]=replace(edges[-1],id=EdgeId(len(edges)+7)); corrupted=replace(snapshot,edges=tuple(edges))
        elif mutation=="missing_vertex":
            loops=list(snapshot.loops); loops[0]=replace(loops[0],vertex_id=VertexId(len(snapshot.vertices)+10)); corrupted=replace(snapshot,loops=tuple(loops))
        else:
            loops=list(snapshot.loops); loops[0]=replace(loops[0],edge_id=EdgeId(len(snapshot.edges)+10)); corrupted=replace(snapshot,loops=tuple(loops))
        assert expected_codes[mutation] in _error_codes(corrupted),f"seed={SEED}, case={case_index}, mutation={mutation}"
def _json_value(rng,depth=0):
    if depth>=4: return rng.choice((None,True,rng.randint(-10_000,10_000),rng.uniform(-1e6,1e6),"Юнікод_日本語"))
    kind=rng.randrange(0,6)
    if kind==0: return [_json_value(rng,depth+1) for _ in range(rng.randrange(0,5))]
    if kind==1: return {f"ключ_{index}":_json_value(rng,depth+1) for index in range(rng.randrange(0,5))}
    return _json_value(rng,4)
def test_seeded_json_contract_roundtrips_and_reports_non_finite_path():
    rng=Random(SEED^0xC0FFEE)
    for _case_index in range(200):
        value=_json_value(rng); validate_json_value(value); encoded=json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False); decoded=json.loads(encoded); validate_json_value(decoded)
    payload={"animations":{"walk":[{"time":0.0},{"time":float("nan")} ]}}
    with pytest.raises(SpineJsonContractError) as captured: validate_json_value(payload)
    assert captured.value.path=="$.animations.walk[1].time"
