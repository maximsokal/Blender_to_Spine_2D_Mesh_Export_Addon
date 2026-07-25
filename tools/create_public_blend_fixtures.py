#!/usr/bin/env python3
"""Create small deterministic public .blend fixtures for Rewrite regression tests."""
from __future__ import annotations
import argparse
from pathlib import Path
import bpy
FIXTURE_NAMES=("procedural_noise","nested_node_groups","overlapping_uv","non_manifold","negative_scale_modifier")
def _reset(): bpy.ops.wm.read_factory_settings(use_empty=True)
def _material(name,color):
    material=bpy.data.materials.new(name); material.use_nodes=True; principled=material.node_tree.nodes.get("Principled BSDF"); principled.inputs["Base Color"].default_value=color; principled.inputs["Roughness"].default_value=0.6; return material
def _object_from_pydata(name,vertices,faces):
    mesh=bpy.data.meshes.new(f"{name}Mesh"); mesh.from_pydata(vertices,(),faces); mesh.update(calc_edges=True); obj=bpy.data.objects.new(name,mesh); bpy.context.scene.collection.objects.link(obj); return obj
def _quad(name="Hero"):
    obj=_object_from_pydata(name,((-1.,-1.,0.),(1.,-1.,0.),(1.,1.,0.),(-1.,1.,0.)),((0,1,2,3),)); uv=obj.data.uv_layers.new(name="UVMap")
    for loop,value in zip(uv.data,((0.,0.),(1.,0.),(1.,1.),(0.,1.)),strict=True): loop.uv=value
    return obj
def _activate(obj):
    for candidate in bpy.context.view_layer.objects: candidate.select_set(False)
    obj.select_set(True); bpy.context.view_layer.objects.active=obj
def _ensure_camera():
    camera_data=bpy.data.cameras.new("Spine2D_FixtureCamera"); camera_data.type="ORTHO"; camera_data.ortho_scale=5.; camera=bpy.data.objects.new("Spine2D_FixtureCamera",camera_data); camera.location=(0.,0.,5.); bpy.context.scene.collection.objects.link(camera); bpy.context.scene.camera=camera
    light_data=bpy.data.lights.new("Spine2D_FixtureLight","AREA"); light_data.energy=1000.; light_data.shape="DISK"; light_data.size=5.; light=bpy.data.objects.new("Spine2D_FixtureLight",light_data); light.location=(0.,0.,4.); bpy.context.scene.collection.objects.link(light)
def _save(path,obj):
    _ensure_camera(); _activate(obj); bpy.context.scene.frame_start=0; bpy.context.scene.frame_end=2; path.parent.mkdir(parents=True,exist_ok=True); result=bpy.ops.wm.save_as_mainfile(filepath=str(path),check_existing=False)
    if "FINISHED" not in result: raise RuntimeError(f"unable to save fixture {path}: {result}")
def procedural_noise(path):
    _reset(); obj=_quad(); material=_material("ProceduralNoiseMaterial",(0.2,0.4,0.8,1.)); nodes=material.node_tree.nodes; links=material.node_tree.links; principled=nodes.get("Principled BSDF"); noise=nodes.new("ShaderNodeTexNoise"); noise.inputs["Scale"].default_value=3.; noise.inputs["Detail"].default_value=2.; ramp=nodes.new("ShaderNodeValToRGB"); ramp.color_ramp.elements[0].color=(0.02,0.1,0.8,1.); ramp.color_ramp.elements[1].color=(1.,0.2,0.02,1.); links.new(noise.outputs["Fac"],ramp.inputs["Fac"]); links.new(ramp.outputs["Color"],principled.inputs["Base Color"]); obj.data.materials.append(material); _save(path,obj)
def nested_node_groups(path):
    _reset(); obj=_quad(); material=_material("NestedGroupMaterial",(0.1,0.8,0.2,1.)); tree=material.node_tree; principled=tree.nodes.get("Principled BSDF"); inner=bpy.data.node_groups.new("Spine2D_InnerColor","ShaderNodeTree"); inner.interface.new_socket(name="Color",in_out="OUTPUT",socket_type="NodeSocketColor"); inner_output=inner.nodes.new("NodeGroupOutput"); rgb=inner.nodes.new("ShaderNodeRGB"); rgb.outputs["Color"].default_value=(0.75,0.08,0.9,1.); inner.links.new(rgb.outputs["Color"],inner_output.inputs["Color"]); outer=bpy.data.node_groups.new("Spine2D_OuterColor","ShaderNodeTree"); outer.interface.new_socket(name="Color",in_out="OUTPUT",socket_type="NodeSocketColor"); outer_output=outer.nodes.new("NodeGroupOutput"); inner_node=outer.nodes.new("ShaderNodeGroup"); inner_node.node_tree=inner; outer.links.new(inner_node.outputs["Color"],outer_output.inputs["Color"]); group_node=tree.nodes.new("ShaderNodeGroup"); group_node.node_tree=outer; tree.links.new(group_node.outputs["Color"],principled.inputs["Base Color"]); obj.data.materials.append(material); _save(path,obj)
def overlapping_uv(path):
    _reset(); obj=_object_from_pydata("Hero",((-2.,-1.,0.),(0.,-1.,0.),(0.,1.,0.),(-2.,1.,0.),(0.,-1.,0.),(2.,-1.,0.),(2.,1.,0.),(0.,1.,0.)),((0,1,2,3),(4,5,6,7))); uv=obj.data.uv_layers.new(name="UVMap")
    for loop,value in zip(uv.data,((0.,0.),(1.,0.),(1.,1.),(0.,1.))*2,strict=True): loop.uv=value
    obj.data.materials.append(_material("LeftMaterial",(0.9,0.1,0.05,1.))); obj.data.materials.append(_material("RightMaterial",(0.05,0.2,0.9,1.))); obj.data.polygons[0].material_index=0; obj.data.polygons[1].material_index=1; _save(path,obj)
def non_manifold(path):
    _reset(); obj=_object_from_pydata("Hero",((-1.,0.,0.),(1.,0.,0.),(0.,1.,0.),(0.,-1.,0.),(0.,0.,1.)),((0,1,2),(1,0,3),(0,1,4))); obj.data.materials.append(_material("NonManifoldMaterial",(0.1,0.7,0.7,1.))); _save(path,obj)
def negative_scale_modifier(path):
    _reset(); obj=_quad(); obj.scale=(-1.,1.5,1.); solidify=obj.modifiers.new("FixtureSolidify","SOLIDIFY"); solidify.thickness=0.05; obj.data.materials.append(_material("NegativeScaleMaterial",(0.9,0.55,0.05,1.))); _save(path,obj)
def create_all(output):
    creators={"procedural_noise":procedural_noise,"nested_node_groups":nested_node_groups,"overlapping_uv":overlapping_uv,"non_manifold":non_manifold,"negative_scale_modifier":negative_scale_modifier}; paths=[]
    for name in FIXTURE_NAMES:
        path=output/f"{name}.blend"; creators[name](path); paths.append(path)
    return tuple(paths)
def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",type=Path,required=True); ns=parser.parse_args()
    for path in create_all(ns.output.resolve(strict=False)): print(path)
if __name__=="__main__": main()
