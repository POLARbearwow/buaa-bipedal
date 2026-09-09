"""Convert the BUAA Q1 URDF subset to an MJCF scene usable by MuJoCo."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def _vec(node: ET.Element | None, attr: str, default: str = "0 0 0") -> str:
    return (node.get(attr, default) if node is not None else default).strip()


def make_mjcf(urdf_path: Path, output_path: Path) -> None:
    root = ET.parse(urdf_path).getroot()
    links = {x.get("name"): x for x in root.findall("link")}
    joints = list(root.findall("joint"))
    parent_joint = {j.find("child").get("link"): j for j in joints if j.find("child") is not None}
    children = {}
    for joint in joints:
        parent, child = joint.find("parent"), joint.find("child")
        if parent is not None and child is not None:
            children.setdefault(parent.get("link"), []).append(joint)
    roots = [name for name in links if name not in parent_joint]
    if len(roots) != 1:
        raise ValueError(f"expected one URDF root, got {roots}")

    mj = ET.Element("mujoco", {"model": "buaa_q1_v1"})
    ET.SubElement(mj, "compiler", {"angle": "radian", "meshdir": "../meshes", "autolimits": "true"})
    ET.SubElement(mj, "option", {"timestep": "0.001", "iterations": "50", "solver": "Newton"})
    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "texture", {"name": "ground_tex", "type": "2d", "builtin": "checker", "width": "512", "height": "512", "rgb1": "0.32 0.34 0.36", "rgb2": "0.16 0.17 0.18"})
    ET.SubElement(asset, "material", {"name": "ground_mat", "texture": "ground_tex", "texrepeat": "40 40", "reflectance": "0.05"})
    for name, rgba in {"blue": "0.1 0.4 0.8 1", "white": "0.9 0.9 0.9 1", "black": "0 0 0 1", "green": "0.25 0.8 0.4 1"}.items():
        ET.SubElement(asset, "material", {"name": name, "rgba": rgba})
    mesh_names = set()
    for link in links.values():
        visual_mesh = link.find("visual/geometry/mesh")
        if visual_mesh is not None and visual_mesh.get("filename"):
            filename = Path(visual_mesh.get("filename")).name
            if filename not in mesh_names:
                ET.SubElement(asset, "mesh", {"name": Path(filename).stem, "file": filename})
                mesh_names.add(filename)

    worldbody = ET.SubElement(mj, "worldbody")
    ET.SubElement(worldbody, "light", {"name": "key_light", "pos": "2 -2 5", "dir": "-0.3 0.3 -1", "directional": "true", "diffuse": "0.85 0.85 0.85", "specular": "0.25 0.25 0.25", "castshadow": "true"})
    ET.SubElement(worldbody, "light", {"name": "fill_light", "pos": "-2 1 3", "dir": "0.4 -0.2 -1", "directional": "true", "diffuse": "0.35 0.38 0.42", "specular": "0.05 0.05 0.05", "castshadow": "false"})
    ET.SubElement(worldbody, "geom", {"name": "ground", "type": "plane", "size": "100 100 0.01", "friction": "1 0.8 0.5", "material": "ground_mat", "contype": "1", "conaffinity": "1"})
    actuators = ET.SubElement(mj, "actuator")
    sensors = ET.SubElement(mj, "sensor")
    ET.SubElement(sensors, "framequat", {"name": "base_quat", "objtype": "body", "objname": "base_link"})
    ET.SubElement(sensors, "gyro", {"name": "body_gyro", "site": "base_imu"})

    def add_body(parent: ET.Element, name: str, joint: ET.Element | None) -> None:
        link = links[name]
        attrs = {"name": name}
        if joint is not None:
            attrs["pos"] = _vec(joint.find("origin"), "xyz")
            if _vec(joint.find("origin"), "rpy") != "0 0 0":
                attrs["euler"] = _vec(joint.find("origin"), "rpy")
        body = ET.SubElement(parent, "body", attrs)
        inertial = link.find("inertial")
        if inertial is not None:
            mass, inertia = inertial.find("mass"), inertial.find("inertia")
            if mass is not None and inertia is not None:
                iattrs = {"mass": mass.get("value"), "pos": _vec(inertial.find("origin"), "xyz")}
                iattrs["fullinertia"] = " ".join(inertia.get(k, "0") for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
                ET.SubElement(body, "inertial", iattrs)
        visual = link.find("visual")
        if visual is not None:
            mesh = visual.find("geometry/mesh")
            if mesh is not None and mesh.get("filename"):
                gattrs = {"type": "mesh", "mesh": Path(mesh.get("filename")).stem, "contype": "0", "conaffinity": "0", "pos": _vec(visual.find("origin"), "xyz")}
                material = visual.find("material")
                if material is not None and material.get("name") in {"blue", "white", "black", "green"}:
                    gattrs["material"] = material.get("name")
                ET.SubElement(body, "geom", gattrs)
        collision = link.find("collision")
        if collision is not None:
            mesh = collision.find("geometry/mesh")
            if mesh is not None and mesh.get("filename"):
                ET.SubElement(body, "geom", {"type": "mesh", "mesh": Path(mesh.get("filename")).stem, "group": "3", "rgba": "0.2 0.2 0.2 0", "pos": _vec(collision.find("origin"), "xyz")})
        if name == "base_link":
            ET.SubElement(body, "site", {"name": "base_imu", "pos": "0 0 0", "size": "0.005"})
        if joint is not None and joint.get("type") == "revolute":
            # Isaac Lab uses the actuator PD damping below; do not add an
            # unrelated MuJoCo joint damping term on top of it.
            jattrs = {"name": joint.get("name"), "type": "hinge", "axis": _vec(joint.find("axis"), "xyz"), "limited": "true"}
            limit = joint.find("limit")
            if limit is not None:
                jattrs["range"] = f"{limit.get('lower', '-3.14')} {limit.get('upper', '3.14')}"
            dynamics = joint.find("dynamics")
            if dynamics is not None and dynamics.get("friction"):
                jattrs["frictionloss"] = dynamics.get("friction")
            ET.SubElement(body, "joint", jattrs)
            effort = "20"
            if limit is not None and limit.get("effort"):
                effort = limit.get("effort")
            # BUAA Q1 Isaac actuator config overrides the URDF's conservative
            # 20 Nm hip-roll limit with 60 Nm.
            if joint.get("name", "").startswith("hip_roll_"):
                effort = "60"
            ET.SubElement(actuators, "motor", {"name": f"{joint.get('name')}_motor", "joint": joint.get("name"), "gear": "1", "ctrlrange": f"-{effort} {effort}", "ctrllimited": "true"})
        for child_joint in children.get(name, []):
            add_body(body, child_joint.find("child").get("link"), child_joint)

    root_name = roots[0]
    root_body = ET.SubElement(worldbody, "body", {"name": root_name, "pos": "0 0 0.45"})
    ET.SubElement(root_body, "freejoint", {"name": "floating_base"})
    add_body_contents = links[root_name]
    inertial = add_body_contents.find("inertial")
    if inertial is not None:
        mass, inertia = inertial.find("mass"), inertial.find("inertia")
        attrs = {"mass": mass.get("value"), "pos": _vec(inertial.find("origin"), "xyz"), "fullinertia": " ".join(inertia.get(k, "0") for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))}
        ET.SubElement(root_body, "inertial", attrs)
    mesh = add_body_contents.find("visual/geometry/mesh")
    if mesh is not None:
        ET.SubElement(root_body, "geom", {"type": "mesh", "mesh": Path(mesh.get("filename")).stem, "contype": "0", "conaffinity": "0", "material": "blue"})
    ET.SubElement(root_body, "site", {"name": "base_imu", "pos": "0 0 0", "size": "0.005"})
    for joint in children.get(root_name, []):
        add_body(root_body, joint.find("child").get("link"), joint)

    defaults = [-0.36, 0.18, 2.35, -0.33, -0.77, 0.36, -0.18, -2.35, 0.33, 0.77]
    keyframe = ET.SubElement(mj, "keyframe")
    ET.SubElement(keyframe, "key", {"name": "default_pos", "qpos": "0 0 0.45 1 0 0 0 " + " ".join(map(str, defaults))})
    ET.indent(mj, space="  ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(mj).write(output_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    make_mjcf(args.urdf, args.output)
