#!/usr/bin/env python3
"""Convert the delivered XHAND1 v1.3 URDF packages to MuJoCo MJCF.

The source delivery is treated as read-only.  Generated meshes, normalized URDFs,
tactile JSON, MJCF components and scenes are written next to this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


SIDES = ("right", "left")
FINGERS = ("thumb", "index", "mid", "ring", "pinky")


def fmt(value: float) -> str:
    if abs(value) < 5e-13:
        value = 0.0
    return f"{value:.10g}"


def vector(text: str | None, default: str = "0 0 0") -> list[float]:
    return [float(item) for item in (text or default).split()]


def vector_text(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(fmt(value) for value in values)


def write_bytes_if_changed(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return
    path.write_bytes(content)


def write_text_if_changed(path: Path, content: str) -> None:
    write_bytes_if_changed(path, content.encode("utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_delivery(explicit: Path | None) -> Path:
    if explicit:
        delivery = explicit.expanduser().resolve()
    else:
        delivery = Path(__file__).resolve().parents[2] / "Xhand1交付资料-带触觉"
    archive = delivery / "urdf" / "Xhand-urdf.zip"
    if not archive.is_file():
        raise FileNotFoundError(f"找不到 URDF 压缩包: {archive}")
    return delivery


def copy_json(source: Path, destination: Path) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    points = payload.get("measurement_points", [])
    if len(points) != 120:
        raise ValueError(f"{source} 应有 120 个触觉点，实际为 {len(points)}")
    if [point.get("point") for point in points] != list(range(1, 121)):
        raise ValueError(f"{source} 的触觉点编号不是 1..120")
    write_text_if_changed(
        destination,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )
    return payload


def tactile_sources(delivery: Path, output: Path) -> dict[str, dict]:
    tactile_dir = delivery / "触觉传感器"
    t16_candidates = sorted(tactile_dir.glob("points_t16_transformed*.json"))
    if not t16_candidates:
        raise FileNotFoundError("找不到 points_t16_transformed JSON")

    sources = {
        "t16": t16_candidates[0],
        "t30_right": tactile_dir / "points_t30_right_hand_transformed.json",
        "t30_left": tactile_dir / "points_t30_left_hand_transformed.json",
    }
    for source in sources.values():
        if not source.is_file():
            raise FileNotFoundError(f"找不到触觉坐标: {source}")

    return {
        key: copy_json(source, output / "tactile" / f"{key}.json")
        for key, source in sources.items()
    }


def extract_urdf_assets(delivery: Path, output: Path) -> dict[str, Path]:
    archive = delivery / "urdf" / "Xhand-urdf.zip"
    urdf_paths: dict[str, Path] = {}

    with zipfile.ZipFile(archive) as package:
        members = package.namelist()
        for side in SIDES:
            urdf_suffix = f"/urdf/xhand_{side}.urdf"
            matches = [name for name in members if name.endswith(urdf_suffix)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"期望在压缩包中找到唯一 {urdf_suffix}，实际: {matches}"
                )

            urdf_content = package.read(matches[0])
            urdf_path = output / "source_urdf" / side / f"xhand_{side}.urdf"
            write_bytes_if_changed(urdf_path, urdf_content)
            urdf_paths[side] = urdf_path

            mesh_marker = f"/xhand1_{side}(1)/meshes/"
            mesh_members = [
                name
                for name in members
                if mesh_marker in name and name.lower().endswith(".stl")
            ]
            if len(mesh_members) != 30:
                raise RuntimeError(
                    f"{side} 手应有 30 个 STL，实际为 {len(mesh_members)}"
                )
            for member in mesh_members:
                write_bytes_if_changed(
                    output / "assets" / side / Path(member).name,
                    package.read(member),
                )

    return urdf_paths


def origin_attributes(element: ET.Element | None) -> dict[str, str]:
    if element is None:
        return {}
    attrs: dict[str, str] = {}
    xyz = vector(element.get("xyz"))
    rpy = vector(element.get("rpy"))
    if any(abs(value) > 1e-12 for value in xyz):
        attrs["pos"] = vector_text(xyz)
    if any(abs(value) > 1e-12 for value in rpy):
        attrs["euler"] = vector_text(rpy)
    return attrs


def mesh_name(side: str, filename: str) -> str:
    return f"{side}_{Path(filename).stem.lower()}"


def mesh_filename(mesh_element: ET.Element) -> str:
    filename = mesh_element.get("filename", "")
    if not filename:
        raise ValueError("URDF mesh 缺少 filename")
    return Path(filename).name


def link_rgba(link: ET.Element) -> str:
    color = link.find("./visual/material/color")
    return color.get("rgba", "0.72 0.74 0.78 1") if color is not None else "0.72 0.74 0.78 1"


def tactile_points_for_side(
    side: str, tactile_data: dict[str, dict]
) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for finger in FINGERS:
        body = (
            f"{side}_hand_thumb_rota_link2"
            if finger == "thumb"
            else f"{side}_hand_{finger}_rota_link2"
            if finger == "index"
            else f"{side}_hand_{finger}_link2"
        )
        source_key = f"t30_{side}" if finger == "thumb" else "t16"
        result[body] = tactile_data[source_key]["measurement_points"]
    return result


def add_inertial(body: ET.Element, link: ET.Element, has_dof: bool) -> None:
    source = link.find("inertial")
    if source is None:
        if has_dof:
            raise ValueError(f"动态 link {link.get('name')} 缺少 inertial")
        return

    mass_element = source.find("mass")
    inertia = source.find("inertia")
    mass = float(mass_element.get("value", "0")) if mass_element is not None else 0.0

    # SolidWorks exported fixed decorative links with singular 1e-11 inertia.
    # They have no DOF, so omitting their negligible inertia is both stable and
    # physically equivalent at simulation precision.
    if not has_dof and mass < 1e-5:
        return
    if inertia is None or mass <= 0:
        if has_dof:
            raise ValueError(f"动态 link {link.get('name')} 的惯量无效")
        return

    attrs = origin_attributes(source.find("origin"))
    attrs["mass"] = fmt(mass)
    attrs["fullinertia"] = vector_text(
        [
            float(inertia.get("ixx", "0")),
            float(inertia.get("iyy", "0")),
            float(inertia.get("izz", "0")),
            float(inertia.get("ixy", "0")),
            float(inertia.get("ixz", "0")),
            float(inertia.get("iyz", "0")),
        ]
    )
    ET.SubElement(body, "inertial", attrs)


def joint_info(joint: ET.Element) -> dict:
    joint_type = joint.get("type")
    info = {
        "name": joint.get("name", ""),
        "type": joint_type,
        "parent": joint.find("parent").get("link"),
        "child": joint.find("child").get("link"),
        "origin": joint.find("origin"),
    }
    if joint_type == "revolute":
        axis = joint.find("axis")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"旋转关节 {info['name']} 缺少 limit")
        info.update(
            axis=(axis.get("xyz") if axis is not None else "1 0 0"),
            lower=float(limit.get("lower")),
            upper=float(limit.get("upper")),
            effort=float(limit.get("effort")),
            velocity=float(limit.get("velocity")),
        )
    return info


def close_targets(joints: list[dict], pose: str) -> list[float]:
    targets: list[float] = []
    for joint in joints:
        name = joint["name"]
        if pose == "open":
            target = 0.0
        elif pose == "pinch":
            if "thumb_bend" in name:
                target = 0.55
            elif "thumb_rota_joint1" in name:
                target = 0.80
            elif "thumb_rota_joint2" in name:
                target = 1.05
            elif "index_bend" in name:
                target = 0.0
            elif "index_joint1" in name:
                target = 1.25
            elif "index_joint2" in name:
                target = 0.95
            else:
                target = 0.20
        else:
            if "thumb_bend" in name:
                target = 0.85
            elif "thumb_rota_joint1" in name:
                target = 0.85
            elif "thumb_rota_joint2" in name:
                target = 1.20
            elif "index_bend" in name:
                target = 0.0
            elif name.endswith("joint1"):
                target = 1.35
            elif name.endswith("joint2"):
                target = 1.15
            else:
                target = 0.0
        target = min(max(target, joint["lower"]), joint["upper"])
        targets.append(target)
    return targets


def build_mjcf(
    side: str,
    urdf_path: Path,
    tactile_data: dict[str, dict],
    destination: Path,
) -> dict:
    urdf_root = ET.parse(urdf_path).getroot()
    links = {link.get("name"): link for link in urdf_root.findall("link")}
    joint_list = [joint_info(joint) for joint in urdf_root.findall("joint")]
    child_links = {joint["child"] for joint in joint_list}
    root_links = [name for name in links if name not in child_links]
    if len(root_links) != 1:
        raise ValueError(f"{side} 手应有唯一根 link，实际: {root_links}")

    children: dict[str, list[dict]] = {name: [] for name in links}
    for joint in joint_list:
        children[joint["parent"]].append(joint)
    revolute_joints = [joint for joint in joint_list if joint["type"] == "revolute"]
    if len(links) != 30 or len(joint_list) != 29 or len(revolute_joints) != 12:
        raise ValueError(
            f"{side} 手结构异常: links={len(links)}, joints={len(joint_list)}, "
            f"revolute={len(revolute_joints)}"
        )

    tactile_by_body = tactile_points_for_side(side, tactile_data)
    root = ET.Element("mujoco", {"model": f"xhand1_{side}"})
    ET.SubElement(
        root,
        "compiler",
        {
            "angle": "radian",
            "meshdir": f"assets/{side}",
            "autolimits": "true",
            "balanceinertia": "true",
            "inertiafromgeom": "false",
            "fusestatic": "false",
        },
    )
    ET.SubElement(
        root,
        "option",
        {
            "timestep": "0.001",
            "integrator": "implicitfast",
            "cone": "elliptic",
            "impratio": "10",
        },
    )
    ET.SubElement(root, "size", {"memory": "128M"})

    defaults = ET.SubElement(root, "default")
    visual_default = ET.SubElement(defaults, "default", {"class": "xhand_visual"})
    ET.SubElement(
        visual_default,
        "geom",
        {"type": "mesh", "contype": "0", "conaffinity": "0", "group": "2", "mass": "0"},
    )
    collision_default = ET.SubElement(defaults, "default", {"class": "xhand_collision"})
    ET.SubElement(
        collision_default,
        "geom",
        {
            "type": "mesh",
            "contype": "1",
            "conaffinity": "1",
            "condim": "4",
            "friction": "1 0.005 0.0001",
            "solref": "0.008 1",
            "solimp": "0.9 0.95 0.001",
            "group": "3",
            "rgba": "0.2 0.7 0.2 0.18",
            "mass": "0",
        },
    )
    taxel_default = ET.SubElement(defaults, "default", {"class": "xhand_taxel"})
    ET.SubElement(
        taxel_default,
        "site",
        {
            "type": "sphere",
            # The delivered points lie on the silicone measurement lattice while
            # MuJoCo contacts lie on the convex hull of the STL.  A 2.5 mm
            # capture radius covers the measured 1.8--2.2 mm hull offset.
            "size": "0.0025",
            "group": "4",
            "rgba": "0.1 1 0.2 0.28",
        },
    )
    joint_default = ET.SubElement(defaults, "default", {"class": "xhand_joint"})
    ET.SubElement(
        joint_default,
        "joint",
        {"damping": "0.02", "armature": "0.0002", "frictionloss": "0.002"},
    )

    assets = ET.SubElement(root, "asset")
    mesh_files: dict[str, str] = {}
    for link in links.values():
        for mesh in link.findall("./visual/geometry/mesh") + link.findall("./collision/geometry/mesh"):
            filename = mesh_filename(mesh)
            mesh_files[mesh_name(side, filename)] = filename
    for name, filename in sorted(mesh_files.items()):
        ET.SubElement(assets, "mesh", {"name": name, "file": filename})

    worldbody = ET.SubElement(root, "worldbody")

    def add_link(
        parent: ET.Element,
        link_name: str,
        incoming_joint: dict | None = None,
    ) -> None:
        body_attrs = {"name": link_name}
        if incoming_joint is not None:
            body_attrs.update(origin_attributes(incoming_joint["origin"]))
        body = ET.SubElement(parent, "body", body_attrs)

        has_dof = incoming_joint is not None and incoming_joint["type"] == "revolute"
        add_inertial(body, links[link_name], has_dof)
        if has_dof:
            ET.SubElement(
                body,
                "joint",
                {
                    "name": incoming_joint["name"],
                    "class": "xhand_joint",
                    "type": "hinge",
                    "axis": incoming_joint["axis"],
                    "range": vector_text([incoming_joint["lower"], incoming_joint["upper"]]),
                    "actuatorfrcrange": vector_text(
                        [-incoming_joint["effort"], incoming_joint["effort"]]
                    ),
                },
            )

        link = links[link_name]
        rgba = link_rgba(link)
        for index, visual in enumerate(link.findall("visual"), start=1):
            mesh = visual.find("./geometry/mesh")
            if mesh is None:
                continue
            attrs = {
                "name": f"{link_name}_visual_{index}",
                "class": "xhand_visual",
                "mesh": mesh_name(side, mesh_filename(mesh)),
                "rgba": rgba,
            }
            attrs.update(origin_attributes(visual.find("origin")))
            ET.SubElement(body, "geom", attrs)

        for index, collision in enumerate(link.findall("collision"), start=1):
            mesh = collision.find("./geometry/mesh")
            if mesh is None:
                continue
            attrs = {
                "name": f"{link_name}_collision_{index}",
                "class": "xhand_collision",
                "mesh": mesh_name(side, mesh_filename(mesh)),
            }
            attrs.update(origin_attributes(collision.find("origin")))
            ET.SubElement(body, "geom", attrs)

        points = tactile_by_body.get(link_name)
        if points:
            finger = next(finger for finger in FINGERS if f"_{finger}_" in f"_{link_name}_")
            for point in points:
                position_m = [float(point[axis]) / 1000.0 for axis in ("x", "y", "z")]
                ET.SubElement(
                    body,
                    "site",
                    {
                        "name": f"{side}_tactile_{finger}_{point['point']:03d}_site",
                        "class": "xhand_taxel",
                        "pos": vector_text(position_m),
                    },
                )

        for child_joint in children[link_name]:
            add_link(body, child_joint["child"], child_joint)

    add_link(worldbody, root_links[0])

    # URDF adjacent links are allowed to overlap around their joint housings.
    # MuJoCo otherwise treats those convex mesh hulls as self contacts.  Exclude
    # every pair of rigid clusters that is identical or directly joint-adjacent,
    # while preserving non-adjacent/inter-finger self collision.
    incoming = {joint["child"]: joint for joint in joint_list}

    def rigid_cluster(link_name: str) -> str:
        while link_name in incoming and incoming[link_name]["type"] == "fixed":
            link_name = incoming[link_name]["parent"]
        return link_name

    adjacent_clusters = {
        frozenset((rigid_cluster(joint["parent"]), rigid_cluster(joint["child"])))
        for joint in revolute_joints
    }
    excluded_pairs: list[tuple[str, str]] = []
    link_names = list(links)
    for index, body1 in enumerate(link_names):
        cluster1 = rigid_cluster(body1)
        for body2 in link_names[index + 1 :]:
            cluster2 = rigid_cluster(body2)
            clusters = frozenset((cluster1, cluster2))
            if cluster1 == cluster2 or clusters in adjacent_clusters:
                excluded_pairs.append((body1, body2))
    contact = ET.SubElement(root, "contact")
    for body1, body2 in excluded_pairs:
        ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})

    actuator = ET.SubElement(root, "actuator")
    for joint in revolute_joints:
        kp = 10.0 if joint["effort"] > 0.5 else 4.0
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{joint['name']}_actuator",
                "joint": joint["name"],
                "kp": fmt(kp),
                "dampratio": "1",
                "ctrlrange": vector_text([joint["lower"], joint["upper"]]),
                "forcerange": vector_text([-joint["effort"], joint["effort"]]),
            },
        )

    sensors = ET.SubElement(root, "sensor")
    for finger in FINGERS:
        for point in range(1, 121):
            base = f"{side}_tactile_{finger}_{point:03d}"
            ET.SubElement(sensors, "touch", {"name": base, "site": f"{base}_site"})

    keyframe = ET.SubElement(root, "keyframe")
    for pose in ("open", "pinch", "close"):
        ET.SubElement(
            keyframe,
            "key",
            {
                "name": f"xhand_{pose}",
                "ctrl": vector_text(close_targets(revolute_joints, pose)),
            },
        )

    ET.indent(root, space="  ")
    write_text_if_changed(destination, ET.tostring(root, encoding="unicode") + "\n")
    return {
        "links": len(links),
        "joints": len(joint_list),
        "actuated_joints": len(revolute_joints),
        "meshes": len(mesh_files),
        "touch_taxels": len(FINGERS) * 120,
        "joint_order": [joint["name"] for joint in revolute_joints],
        "joint_velocity_limits": [joint["velocity"] for joint in revolute_joints],
    }


def build_scene(side: str, destination: Path) -> None:
    root = ET.Element("mujoco", {"model": f"xhand1_{side}_scene"})
    ET.SubElement(root, "include", {"file": f"xhand_{side}.xml"})
    ET.SubElement(root, "statistic", {"center": "0 0 0.09", "extent": "0.24"})

    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.65 0.65 0.65", "ambient": "0.25 0.25 0.25", "specular": "0 0 0"},
    )
    ET.SubElement(visual, "rgba", {"contactpoint": "1 0.2 0.2 1", "contactforce": "1 0.8 0.2 1"})
    ET.SubElement(visual, "scale", {"forcewidth": "0.004", "contactwidth": "0.003"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "xhand_ground_texture",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.18 0.23 0.28",
            "rgb2": "0.08 0.11 0.14",
            "width": "256",
            "height": "256",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "xhand_ground", "texture": "xhand_ground_texture", "texrepeat": "5 5", "reflectance": "0.15"},
    )

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "-0.2 -0.3 0.5", "dir": "0.3 0.4 -1", "directional": "true"})
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": "0 0 -0.015",
            "size": "0 0 0.02",
            "material": "xhand_ground",
            "condim": "4",
            "friction": "1 0.005 0.0001",
        },
    )
    lateral = 0.0405 if side == "right" else -0.0405
    object_body = ET.SubElement(
        worldbody,
        "body",
        {"name": "grasp_object", "pos": vector_text([0.071, lateral, 0.101])},
    )
    ET.SubElement(object_body, "freejoint", {"name": "grasp_object_free"})
    ET.SubElement(
        object_body,
        "inertial",
        {"pos": "0 0 0", "mass": "0.03", "diaginertia": "3.468e-6 3.468e-6 3.468e-6"},
    )
    ET.SubElement(
        object_body,
        "geom",
        {
            "name": "grasp_object_geom",
            "type": "sphere",
            "size": "0.017",
            "mass": "0",
            "rgba": "0.95 0.45 0.08 1",
            "condim": "4",
            "friction": "1.1 0.01 0.0002",
            "solref": "0.008 1",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "object_support",
            "type": "cylinder",
            "pos": vector_text([0.071, lateral, 0.037]),
            "size": "0.006 0.047",
            "rgba": "0.32 0.38 0.44 1",
            "condim": "4",
            "friction": "1.2 0.005 0.0001",
        },
    )
    probe = ET.SubElement(
        worldbody,
        "body",
        {"name": "tactile_probe", "mocap": "true", "pos": "0 0 -1"},
    )
    ET.SubElement(
        probe,
        "geom",
        {
            "name": "tactile_probe_geom",
            "type": "sphere",
            "size": "0.0018",
            "rgba": "0.1 0.8 1 0.75",
            "mass": "0",
            "condim": "3",
            "friction": "0.8 0.005 0.0001",
        },
    )

    ET.indent(root, space="  ")
    write_text_if_changed(destination, ET.tostring(root, encoding="unicode") + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="将 XHAND1 v1.3 URDF/STL/触觉坐标生成 MuJoCo MJCF"
    )
    parser.add_argument(
        "--delivery",
        type=Path,
        help="Xhand1交付资料-带触觉 目录（默认自动定位）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="输出目录（默认为脚本所在目录）",
    )
    args = parser.parse_args()

    delivery = find_delivery(args.delivery)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    urdf_paths = extract_urdf_assets(delivery, output)
    tactile_data = tactile_sources(delivery, output)
    summary = {}
    for side in SIDES:
        summary[side] = build_mjcf(
            side,
            urdf_paths[side],
            tactile_data,
            output / f"xhand_{side}.xml",
        )
        build_scene(side, output / f"scene_{side}.xml")

    archive = delivery / "urdf" / "Xhand-urdf.zip"
    manifest = {
        "source": {
            "delivery_directory": str(delivery),
            "urdf_archive": str(archive),
            "urdf_archive_sha256": sha256(archive),
            "urdf_version": "1.3",
        },
        "units": {"mesh": "metre", "tactile_input": "millimetre", "mjcf": "metre"},
        "models": summary,
        "notes": [
            "原始交付目录只读，未删除或改写任何源文件。",
            "退化的固定装饰 link 惯量被忽略，12 个运动 link 惯量保留。",
            "触觉 site 来自 transformed JSON，挂载在各指 link2 局部坐标系。",
        ],
    }
    write_text_if_changed(
        output / "conversion_manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )

    print(f"转换完成: {output}")
    for side in SIDES:
        info = summary[side]
        print(
            f"  {side}: {info['links']} links, {info['actuated_joints']} actuators, "
            f"{info['meshes']} meshes, {info['touch_taxels']} tactile taxels"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"转换失败: {error}", file=sys.stderr)
        raise SystemExit(1)
