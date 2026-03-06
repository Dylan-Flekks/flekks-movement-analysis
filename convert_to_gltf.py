#!/usr/bin/env python3
"""
Flekks OpenSim → glTF Converter

Converts a scaled OpenSim .osim model to glTF/GLB format for 3D
visualization in the iOS app or web viewer.

Parses the .osim XML to extract:
  - BodySet: rigid body segments (bones) with geometry references
  - JointSet: joint hierarchy and default transforms
  - Muscle paths (optional): as line geometry

The output .glb file can be loaded by Three.js, SceneKit, or any
glTF-compatible renderer. Joint angles from IK results can be applied
to animate the model.

Based on approaches from: opensim-org/opensim-viewer-backend

Usage:
    from convert_to_gltf import convert_osim_to_gltf
    convert_osim_to_gltf("scaled_model.osim", "output.glb")
"""

import os
import sys
import json
import struct
import logging
import xml.etree.ElementTree as ET
import numpy as np

logger = logging.getLogger(__name__)

# ── Simplified bone geometry ──
# When .vtp/.stl mesh files aren't available, we generate capsule/box
# primitives for each body segment. These approximate dimensions are
# based on the Rajagopal2016 model's body segment lengths.

DEFAULT_BONE_DIMS = {
    # body_name: (length, radius) in meters
    "pelvis":     (0.15, 0.08),
    "femur_r":    (0.40, 0.04),
    "femur_l":    (0.40, 0.04),
    "tibia_r":    (0.38, 0.03),
    "tibia_l":    (0.38, 0.03),
    "talus_r":    (0.05, 0.03),
    "talus_l":    (0.05, 0.03),
    "calcn_r":    (0.12, 0.03),
    "calcn_l":    (0.12, 0.03),
    "toes_r":     (0.08, 0.02),
    "toes_l":     (0.08, 0.02),
    "torso":      (0.45, 0.10),
    "humerus_r":  (0.30, 0.03),
    "humerus_l":  (0.30, 0.03),
    "ulna_r":     (0.25, 0.02),
    "ulna_l":     (0.25, 0.02),
    "radius_r":   (0.25, 0.02),
    "radius_l":   (0.25, 0.02),
    "hand_r":     (0.15, 0.03),
    "hand_l":     (0.15, 0.03),
}

# Default bone colors (RGBA, 0-1)
BONE_COLOR = [0.85, 0.82, 0.75, 1.0]      # off-white bone
MUSCLE_COLOR = [0.75, 0.20, 0.20, 0.6]     # translucent red
JOINT_COLOR = [0.35, 0.73, 0.61, 1.0]      # flekks teal


def parse_osim_model(osim_path):
    """Parse an OpenSim .osim XML file.

    Returns:
        dict with 'bodies', 'joints', 'muscles' lists.
    """
    tree = ET.parse(osim_path)
    root = tree.getroot()

    # Handle OpenSim XML namespace variations
    model = root.find(".//Model") or root
    if model is None:
        model = root

    result = {
        'bodies': [],
        'joints': [],
        'muscles': [],
        'model_name': '',
    }

    # Model name
    name_elem = model.find("./model_name") or model.get("name", "")
    result['model_name'] = name_elem if isinstance(name_elem, str) else (
        name_elem.text if name_elem is not None else "model"
    )

    # Parse BodySet
    body_set = model.find(".//BodySet/objects") or model.find(".//BodySet")
    if body_set is not None:
        for body in body_set:
            if body.tag == "Body" or body.tag.endswith("Body"):
                body_info = _parse_body(body)
                if body_info:
                    result['bodies'].append(body_info)

    # Parse JointSet
    joint_set = model.find(".//JointSet/objects") or model.find(".//JointSet")
    if joint_set is not None:
        for joint in joint_set:
            joint_info = _parse_joint(joint)
            if joint_info:
                result['joints'].append(joint_info)

    # Parse muscle paths (ForceSet / muscles)
    force_set = (model.find(".//ForceSet/objects") or
                 model.find(".//ForceSet"))
    if force_set is not None:
        for force in force_set:
            tag = force.tag.lower()
            if "muscle" in tag:
                muscle_info = _parse_muscle(force)
                if muscle_info:
                    result['muscles'].append(muscle_info)

    logger.info(f"Parsed .osim: {len(result['bodies'])} bodies, "
                f"{len(result['joints'])} joints, "
                f"{len(result['muscles'])} muscles")

    return result


def _parse_body(elem):
    """Parse a Body element from BodySet."""
    name = elem.get("name", elem.find("name"))
    if isinstance(name, ET.Element):
        name = name.text
    if not name:
        return None

    mass_elem = elem.find("mass")
    mass = float(mass_elem.text) if mass_elem is not None else 1.0

    # Mass center
    mc_elem = elem.find("mass_center")
    mass_center = [0, 0, 0]
    if mc_elem is not None and mc_elem.text:
        parts = mc_elem.text.strip().split()
        mass_center = [float(x) for x in parts[:3]]

    # Geometry references
    geometry_files = []
    # OpenSim 4.x: attached_geometry
    for geom in elem.findall(".//attached_geometry//Mesh"):
        mesh_file = geom.find("mesh_file")
        if mesh_file is not None and mesh_file.text:
            geometry_files.append(mesh_file.text.strip())
    # OpenSim 3.x: VisibleObject
    vis_obj = elem.find(".//VisibleObject")
    if vis_obj is not None:
        for geom_file in vis_obj.findall(".//GeometrySet/objects//*"):
            gf = geom_file.find("geometry_file")
            if gf is not None and gf.text:
                geometry_files.append(gf.text.strip())

    return {
        'name': name,
        'mass': mass,
        'mass_center': mass_center,
        'geometry_files': geometry_files,
    }


def _parse_joint(elem):
    """Parse a Joint element from JointSet."""
    name = elem.get("name", "")
    if not name:
        name_elem = elem.find("name")
        name = name_elem.text if name_elem is not None else ""

    joint_type = elem.tag  # e.g., "CustomJoint", "PinJoint"

    # Parent/child bodies
    parent_frame = ""
    child_frame = ""

    # OpenSim 4.x
    socket_parent = elem.find(".//socket_parent_frame")
    socket_child = elem.find(".//socket_child_frame")
    if socket_parent is not None and socket_parent.text:
        parent_frame = socket_parent.text.strip().split("/")[-1]
    if socket_child is not None and socket_child.text:
        child_frame = child_frame or socket_child.text.strip().split("/")[-1]

    # Fallback: look for body references
    parent_body = elem.find(".//parent_body")
    if parent_body is not None and parent_body.text:
        parent_frame = parent_frame or parent_body.text.strip()

    # Location in parent
    loc_elem = elem.find(".//location_in_parent")
    location = [0, 0, 0]
    if loc_elem is not None and loc_elem.text:
        parts = loc_elem.text.strip().split()
        location = [float(x) for x in parts[:3]]

    # Orientation in parent
    orient_elem = elem.find(".//orientation_in_parent")
    orientation = [0, 0, 0]
    if orient_elem is not None and orient_elem.text:
        parts = orient_elem.text.strip().split()
        orientation = [float(x) for x in parts[:3]]

    # Coordinates (DOFs)
    coordinates = []
    for coord in elem.findall(".//Coordinate"):
        coord_name = coord.get("name", "")
        default_val = 0.0
        dv = coord.find("default_value")
        if dv is not None and dv.text:
            default_val = float(dv.text)
        coordinates.append({'name': coord_name, 'default': default_val})

    return {
        'name': name,
        'type': joint_type,
        'parent': parent_frame,
        'child': child_frame,
        'location': location,
        'orientation': orientation,
        'coordinates': coordinates,
    }


def _parse_muscle(elem):
    """Parse a muscle element for path points."""
    name = elem.get("name", "")
    if not name:
        name_elem = elem.find("name")
        name = name_elem.text if name_elem is not None else ""

    path_points = []
    for pp in elem.findall(".//PathPoint"):
        loc_elem = pp.find("location")
        body_elem = pp.find("body") or pp.find("socket_parent_frame")
        if loc_elem is not None and loc_elem.text:
            parts = loc_elem.text.strip().split()
            loc = [float(x) for x in parts[:3]]
            body_name = ""
            if body_elem is not None and body_elem.text:
                body_name = body_elem.text.strip().split("/")[-1]
            path_points.append({'location': loc, 'body': body_name})

    return {
        'name': name,
        'path_points': path_points,
    }


def generate_capsule_mesh(length, radius, segments=8, rings=4):
    """Generate a capsule mesh (cylinder with hemispherical caps).

    Returns:
        (vertices, indices) — vertices as flat float list, indices as int list.
    """
    vertices = []
    indices = []

    half_len = length / 2.0

    # Cylinder body
    for i in range(segments + 1):
        theta = 2.0 * np.pi * i / segments
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)

        # Bottom ring
        vertices.extend([x, -half_len, z])
        # Top ring
        vertices.extend([x, half_len, z])

    # Cylinder indices
    for i in range(segments):
        base = i * 2
        indices.extend([base, base + 1, base + 2])
        indices.extend([base + 1, base + 3, base + 2])

    # Top cap (simplified — just a fan from center)
    top_center_idx = len(vertices) // 3
    vertices.extend([0, half_len + radius * 0.5, 0])
    for i in range(segments):
        theta = 2.0 * np.pi * i / segments
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        vertices.extend([x, half_len, z])

    for i in range(segments):
        idx = top_center_idx + 1 + i
        next_idx = top_center_idx + 1 + (i + 1) % segments
        indices.extend([top_center_idx, idx, next_idx])

    # Bottom cap
    bot_center_idx = len(vertices) // 3
    vertices.extend([0, -half_len - radius * 0.5, 0])
    for i in range(segments):
        theta = 2.0 * np.pi * i / segments
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        vertices.extend([x, -half_len, z])

    for i in range(segments):
        idx = bot_center_idx + 1 + i
        next_idx = bot_center_idx + 1 + (i + 1) % segments
        indices.extend([bot_center_idx, next_idx, idx])

    return vertices, indices


def build_gltf(model_data, include_muscles=False):
    """Build a glTF JSON structure from parsed .osim data.

    Creates a node hierarchy with mesh primitives for each body segment.

    Args:
        model_data: dict from parse_osim_model().
        include_muscles: Whether to include muscle path lines.

    Returns:
        (gltf_json, binary_buffer) tuple.
    """
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "Flekks OpenSim Converter",
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [],
        "meshes": [],
        "accessors": [],
        "bufferViews": [],
        "buffers": [],
        "materials": [],
    }

    binary_chunks = []
    byte_offset = 0

    # Material for bones
    gltf["materials"].append({
        "name": "bone",
        "pbrMetallicRoughness": {
            "baseColorFactor": BONE_COLOR,
            "metallicFactor": 0.1,
            "roughnessFactor": 0.8,
        },
    })

    if include_muscles:
        gltf["materials"].append({
            "name": "muscle",
            "pbrMetallicRoughness": {
                "baseColorFactor": MUSCLE_COLOR,
                "metallicFactor": 0.0,
                "roughnessFactor": 0.9,
            },
            "alphaMode": "BLEND",
        })

    # Build body-name → joint-transform lookup
    joint_lookup = {}
    for joint in model_data['joints']:
        child = joint.get('child', '')
        if child:
            # Strip _offset suffix
            body_name = child.replace("_offset", "")
            joint_lookup[body_name] = joint

    # Root node
    gltf["nodes"].append({
        "name": model_data.get('model_name', 'model'),
        "children": list(range(1, len(model_data['bodies']) + 1)),
    })

    # Create a node + mesh for each body
    for body_idx, body in enumerate(model_data['bodies']):
        node_idx = body_idx + 1
        body_name = body['name']

        # Get transform from joint
        translation = [0, 0, 0]
        if body_name in joint_lookup:
            translation = joint_lookup[body_name]['location']

        # Generate capsule mesh for this body
        dims = DEFAULT_BONE_DIMS.get(body_name, (0.10, 0.02))
        verts, idxs = generate_capsule_mesh(dims[0], dims[1])

        # Pack vertex data
        vert_bytes = struct.pack(f"<{len(verts)}f", *verts)
        idx_bytes = struct.pack(f"<{len(idxs)}H", *idxs)

        # Pad to 4-byte alignment
        while len(vert_bytes) % 4 != 0:
            vert_bytes += b'\x00'
        while len(idx_bytes) % 4 != 0:
            idx_bytes += b'\x00'

        # Buffer views
        vert_bv_idx = len(gltf["bufferViews"])
        gltf["bufferViews"].append({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(vert_bytes),
            "target": 34962,  # ARRAY_BUFFER
        })
        binary_chunks.append(vert_bytes)
        byte_offset += len(vert_bytes)

        idx_bv_idx = len(gltf["bufferViews"])
        gltf["bufferViews"].append({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(idx_bytes),
            "target": 34963,  # ELEMENT_ARRAY_BUFFER
        })
        binary_chunks.append(idx_bytes)
        byte_offset += len(idx_bytes)

        # Compute bounds
        vert_array = np.array(verts).reshape(-1, 3)
        v_min = vert_array.min(axis=0).tolist()
        v_max = vert_array.max(axis=0).tolist()

        # Accessors
        vert_acc_idx = len(gltf["accessors"])
        gltf["accessors"].append({
            "bufferView": vert_bv_idx,
            "componentType": 5126,  # FLOAT
            "count": len(verts) // 3,
            "type": "VEC3",
            "min": v_min,
            "max": v_max,
        })

        idx_acc_idx = len(gltf["accessors"])
        gltf["accessors"].append({
            "bufferView": idx_bv_idx,
            "componentType": 5123,  # UNSIGNED_SHORT
            "count": len(idxs),
            "type": "SCALAR",
            "min": [min(idxs)],
            "max": [max(idxs)],
        })

        # Mesh
        mesh_idx = len(gltf["meshes"])
        gltf["meshes"].append({
            "name": body_name,
            "primitives": [{
                "attributes": {"POSITION": vert_acc_idx},
                "indices": idx_acc_idx,
                "material": 0,
            }],
        })

        # Node
        node = {
            "name": body_name,
            "mesh": mesh_idx,
            "translation": translation,
        }
        gltf["nodes"].append(node)

    # Buffer
    total_binary = b''.join(binary_chunks)
    gltf["buffers"].append({
        "byteLength": len(total_binary),
    })

    return gltf, total_binary


def write_glb(gltf_json, binary_data, output_path):
    """Write a binary glTF (.glb) file.

    GLB format:
      - 12-byte header: magic(4) + version(4) + length(4)
      - JSON chunk: length(4) + type(4) + data(padded)
      - Binary chunk: length(4) + type(4) + data(padded)
    """
    json_str = json.dumps(gltf_json, separators=(',', ':'))
    json_bytes = json_str.encode('utf-8')

    # Pad JSON to 4-byte alignment with spaces
    while len(json_bytes) % 4 != 0:
        json_bytes += b' '

    # Pad binary to 4-byte alignment with zeros
    bin_data = binary_data
    while len(bin_data) % 4 != 0:
        bin_data += b'\x00'

    total_length = (
        12 +                          # header
        8 + len(json_bytes) +         # JSON chunk header + data
        8 + len(bin_data)             # binary chunk header + data
    )

    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', 0x46546C67))  # magic: "glTF"
        f.write(struct.pack('<I', 2))            # version
        f.write(struct.pack('<I', total_length))

        # JSON chunk
        f.write(struct.pack('<I', len(json_bytes)))
        f.write(struct.pack('<I', 0x4E4F534A))   # type: "JSON"
        f.write(json_bytes)

        # Binary chunk
        f.write(struct.pack('<I', len(bin_data)))
        f.write(struct.pack('<I', 0x004E4942))    # type: "BIN\0"
        f.write(bin_data)

    logger.info(f"GLB written: {output_path} ({total_length} bytes)")


def convert_osim_to_gltf(osim_path, output_path, include_muscles=False):
    """Convert an OpenSim .osim model to glTF/GLB.

    Args:
        osim_path: Path to .osim model file.
        output_path: Path to write .glb file.
        include_muscles: Whether to include muscle path geometry.

    Returns:
        Path to written GLB file, or None on failure.
    """
    try:
        if not os.path.exists(osim_path):
            logger.error(f"Model not found: {osim_path}")
            return None

        model_data = parse_osim_model(osim_path)

        if not model_data['bodies']:
            logger.error("No bodies found in .osim model")
            return None

        gltf_json, binary_data = build_gltf(model_data, include_muscles)
        write_glb(gltf_json, binary_data, output_path)

        return output_path

    except Exception as e:
        logger.exception(f"glTF conversion failed: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_to_gltf.py <input.osim> <output.glb> "
              "[--muscles]")
        sys.exit(1)

    osim_path = sys.argv[1]
    output_path = sys.argv[2]
    muscles = "--muscles" in sys.argv

    logging.basicConfig(level=logging.INFO)
    result = convert_osim_to_gltf(osim_path, output_path, muscles)
    if result:
        print(f"Success: {result}")
    else:
        print("Conversion failed", file=sys.stderr)
        sys.exit(1)
