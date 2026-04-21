"""Build binary MRC masks from graph topology and decoupled radii inputs.

Coordinate convention:
- All coordinates are physical coordinates in Angstrom (x, y, z).
- Grid shape is in (Z, Y, X).
- Voxel centers are defined by:
    x = origin_x + ix * voxel_size_x
    y = origin_y + iy * voxel_size_y
    z = origin_z + iz * voxel_size_z

Input decoupling:
- topology: node IDs, node coordinates, edge IDs, edge endpoints
- node_radii: node_id -> radius_A
- edge_radii: edge_id -> radius_A
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import yaml

from mrc_pointcloud import write_mrc


Vec3 = Tuple[float, float, float]
Shape3 = Tuple[int, int, int]


def _load_json_or_dict(path_or_dict: str | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(path_or_dict, Mapping):
        return dict(path_or_dict)
    with open(path_or_dict, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_vec3(values: Sequence[float], name: str) -> Vec3:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must have exactly 3 values.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return float(arr[0]), float(arr[1]), float(arr[2])


def _normalize_shape3(values: Sequence[int], name: str) -> Shape3:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size != 3:
        raise ValueError(f"{name} must have exactly 3 integer values (Z,Y,X).")
    if np.any(arr <= 0):
        raise ValueError(f"{name} values must be positive.")
    return int(arr[0]), int(arr[1]), int(arr[2])


def _normalize_radius_map(raw: Mapping[str, Any], name: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in raw.items():
        r = float(v)
        if not np.isfinite(r) or r < 0:
            raise ValueError(f"{name}[{k}] must be a finite non-negative radius.")
        out[str(k)] = r
    return out


def load_graph_topology(path_or_dict: str | Mapping[str, Any]) -> Dict[str, Any]:
    """Load topology with nodes and edges from JSON path or dict."""
    data = _load_json_or_dict(path_or_dict)
    if "nodes" not in data or "edges" not in data:
        raise ValueError("Topology must contain 'nodes' and 'edges'.")
    if not isinstance(data["nodes"], list) or not isinstance(data["edges"], list):
        raise ValueError("'nodes' and 'edges' must be lists.")

    nodes_out = []
    seen_node_ids = set()
    for node in data["nodes"]:
        if not isinstance(node, Mapping):
            raise ValueError("Each node entry must be an object.")
        if "id" not in node or "coord" not in node:
            raise ValueError("Each node must contain 'id' and 'coord'.")
        node_id = str(node["id"])
        if node_id in seen_node_ids:
            raise ValueError(f"Duplicate node id: {node_id}")
        coord = _normalize_vec3(node["coord"], f"node[{node_id}].coord")
        nodes_out.append({"id": node_id, "coord": coord})
        seen_node_ids.add(node_id)

    edges_out = []
    seen_edge_ids = set()
    for edge in data["edges"]:
        if not isinstance(edge, Mapping):
            raise ValueError("Each edge entry must be an object.")
        if "id" not in edge or "source" not in edge or "target" not in edge:
            raise ValueError("Each edge must contain 'id', 'source', and 'target'.")
        edge_id = str(edge["id"])
        if edge_id in seen_edge_ids:
            raise ValueError(f"Duplicate edge id: {edge_id}")
        source = str(edge["source"])
        target = str(edge["target"])
        edges_out.append({"id": edge_id, "source": source, "target": target})
        seen_edge_ids.add(edge_id)

    return {"nodes": nodes_out, "edges": edges_out}


def load_node_radii(path_or_dict: str | Mapping[str, Any]) -> Dict[str, float]:
    """Load node radius map: node_id -> radius_A."""
    data = _load_json_or_dict(path_or_dict)
    if not isinstance(data, Mapping):
        raise ValueError("Node radii must be a mapping of node_id -> radius_A.")
    return _normalize_radius_map(data, "node_radii")


def load_edge_radii(path_or_dict: str | Mapping[str, Any]) -> Dict[str, float]:
    """Load edge radius map: edge_id -> radius_A."""
    data = _load_json_or_dict(path_or_dict)
    if not isinstance(data, Mapping):
        raise ValueError("Edge radii must be a mapping of edge_id -> radius_A.")
    return _normalize_radius_map(data, "edge_radii")


def normalize_grid_config(grid_config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize grid config.

    Required fields:
    - box_shape_zyx: (Z,Y,X)
    - voxel_size_xyz_A: (x,y,z)
    - origin_xyz_A: (x,y,z)
    """
    if not isinstance(grid_config, Mapping):
        raise ValueError("grid_config must be an object.")
    if "box_shape_zyx" not in grid_config:
        raise ValueError("grid_config missing box_shape_zyx.")
    if "voxel_size_xyz_A" not in grid_config:
        raise ValueError("grid_config missing voxel_size_xyz_A.")
    if "origin_xyz_A" not in grid_config:
        raise ValueError("grid_config missing origin_xyz_A.")

    box_shape_zyx = _normalize_shape3(grid_config["box_shape_zyx"], "box_shape_zyx")
    voxel_size_xyz_A = _normalize_vec3(grid_config["voxel_size_xyz_A"], "voxel_size_xyz_A")
    origin_xyz_A = _normalize_vec3(grid_config["origin_xyz_A"], "origin_xyz_A")
    if any(v <= 0 for v in voxel_size_xyz_A):
        raise ValueError("voxel_size_xyz_A values must be positive.")

    return {
        "box_shape_zyx": box_shape_zyx,
        "voxel_size_xyz_A": voxel_size_xyz_A,
        "origin_xyz_A": origin_xyz_A,
    }


def validate_graph_inputs(
    topology: Mapping[str, Any], node_radii: Mapping[str, float], edge_radii: Mapping[str, float]
) -> Dict[str, Any]:
    """Cross-validate topology and decoupled radii maps."""
    if "nodes" not in topology or "edges" not in topology:
        raise ValueError("topology must contain 'nodes' and 'edges'.")

    node_id_to_coord: Dict[str, np.ndarray] = {}
    for node in topology["nodes"]:
        node_id = str(node["id"])
        coord = np.asarray(node["coord"], dtype=np.float64)
        if coord.shape != (3,) or not np.isfinite(coord).all():
            raise ValueError(f"Invalid node coord for {node_id}.")
        node_id_to_coord[node_id] = coord

    missing_node_radii = sorted([nid for nid in node_id_to_coord if nid not in node_radii])
    if missing_node_radii:
        raise ValueError(f"Missing node radii for node IDs: {missing_node_radii}")

    edge_records = []
    edge_ids = set()
    for edge in topology["edges"]:
        edge_id = str(edge["id"])
        source = str(edge["source"])
        target = str(edge["target"])
        if source not in node_id_to_coord or target not in node_id_to_coord:
            raise ValueError(f"Edge {edge_id} references unknown node(s): {source}, {target}")
        if edge_id in edge_ids:
            raise ValueError(f"Duplicate edge id: {edge_id}")
        edge_ids.add(edge_id)
        edge_records.append({"id": edge_id, "source": source, "target": target})

    missing_edge_radii = sorted([eid for eid in edge_ids if eid not in edge_radii])
    if missing_edge_radii:
        raise ValueError(f"Missing edge radii for edge IDs: {missing_edge_radii}")

    extra_node_radii = sorted([nid for nid in node_radii if nid not in node_id_to_coord])
    extra_edge_radii = sorted([eid for eid in edge_radii if eid not in edge_ids])

    return {
        "node_id_to_coord": node_id_to_coord,
        "node_radii": {str(k): float(v) for k, v in node_radii.items()},
        "edge_radii": {str(k): float(v) for k, v in edge_radii.items()},
        "edges": edge_records,
        "warnings": {
            "extra_node_radii": extra_node_radii,
            "extra_edge_radii": extra_edge_radii,
        },
    }


def _voxel_centers(grid_config: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_len, y_len, x_len = grid_config["box_shape_zyx"]
    vx, vy, vz = grid_config["voxel_size_xyz_A"]
    ox, oy, oz = grid_config["origin_xyz_A"]

    x = ox + np.arange(x_len, dtype=np.float64) * vx
    y = oy + np.arange(y_len, dtype=np.float64) * vy
    z = oz + np.arange(z_len, dtype=np.float64) * vz
    return x, y, z


def _sphere_mask_for_node(
    center_xyz: np.ndarray, radius: float, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    cx, cy, cz = float(center_xyz[0]), float(center_xyz[1]), float(center_xyz[2])
    r2 = float(radius) * float(radius)

    xx = (x - cx) ** 2
    yy = (y - cy) ** 2
    zz = (z - cz) ** 2

    return (zz[:, None, None] + yy[None, :, None] + xx[None, None, :]) <= r2


def _cylinder_mask_for_edge(
    p0_xyz: np.ndarray, p1_xyz: np.ndarray, radius: float, x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> np.ndarray:
    seg = p1_xyz - p0_xyz
    seg_len2 = float(np.dot(seg, seg))
    if seg_len2 == 0.0:
        # Degenerate edge: treat as a sphere at the node center.
        return _sphere_mask_for_node(p0_xyz, radius, x, y, z)

    X, Y, Z = np.meshgrid(x, y, z, indexing="xy")
    P = np.stack([X, Y, Z], axis=-1)  # (Y,X,Z,3)
    # Reorder to (Z,Y,X,3) to align with mask layout.
    P = np.transpose(P, (2, 0, 1, 3))

    W = P - p0_xyz
    t = np.sum(W * seg, axis=-1) / seg_len2
    t_clamped = np.clip(t, 0.0, 1.0)
    closest = p0_xyz + t_clamped[..., None] * seg
    dist2 = np.sum((P - closest) ** 2, axis=-1)
    return dist2 <= (float(radius) * float(radius))


def graph_to_mask(
    grid_config: Mapping[str, Any],
    topology: Mapping[str, Any],
    node_radii: Mapping[str, float],
    edge_radii: Mapping[str, float],
    dtype: Any = np.uint8,
) -> np.ndarray:
    """Construct binary mask (Z,Y,X) from sphere/cylinder unions."""
    grid = normalize_grid_config(grid_config)
    validated = validate_graph_inputs(topology, node_radii, edge_radii)

    x, y, z = _voxel_centers(grid)
    mask = np.zeros(grid["box_shape_zyx"], dtype=bool)

    for node_id, center in validated["node_id_to_coord"].items():
        radius = validated["node_radii"][node_id]
        if radius == 0.0:
            continue
        mask |= _sphere_mask_for_node(center, radius, x, y, z)

    for edge in validated["edges"]:
        edge_id = edge["id"]
        radius = validated["edge_radii"][edge_id]
        if radius == 0.0:
            continue
        p0 = validated["node_id_to_coord"][edge["source"]]
        p1 = validated["node_id_to_coord"][edge["target"]]
        mask |= _cylinder_mask_for_edge(p0, p1, radius, x, y, z)

    return mask.astype(dtype)


def mask_to_mrc(
    output_path: str,
    mask: np.ndarray,
    voxel_size_xyz_A: Sequence[float],
    origin_xyz_A: Sequence[float],
    overwrite: bool = True,
) -> None:
    """Write binary mask to MRC using physical metadata."""
    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise ValueError("mask must be 3D with shape (Z, Y, X).")
    unique_vals = np.unique(arr)
    if np.any((unique_vals != 0) & (unique_vals != 1)):
        raise ValueError("mask must contain only 0/1 values.")
    write_mrc(
        path=output_path,
        data=arr.astype(np.float32),
        voxel_size=tuple(float(v) for v in voxel_size_xyz_A),
        origin=tuple(float(v) for v in origin_xyz_A),
        overwrite=overwrite,
    )


def _load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError("Config must be a YAML object at top level.")
    return loaded


def _require(cfg: Mapping[str, Any], key: str, task: str) -> Any:
    if key not in cfg:
        raise ValueError(f"Missing required key '{key}' for task '{task}'.")
    return cfg[key]


def _template_topology() -> Dict[str, Any]:
    return {
        "nodes": [
            {"id": "N1", "coord": [20.0, 30.0, 25.0]},
            {"id": "N2", "coord": [40.0, 30.0, 25.0]},
            {"id": "N3", "coord": [30.0, 45.0, 25.0]},
        ],
        "edges": [
            {"id": "E1", "source": "N1", "target": "N2"},
            {"id": "E2", "source": "N2", "target": "N3"},
        ],
    }


def _template_node_radii() -> Dict[str, float]:
    return {"N1": 4.0, "N2": 5.0, "N3": 4.0}


def _template_edge_radii() -> Dict[str, float]:
    return {"E1": 2.0, "E2": 2.0}


def _template_grid_config() -> Dict[str, Any]:
    return {
        "box_shape_zyx": [80, 80, 80],
        "voxel_size_xyz_A": [1.0, 1.0, 1.0],
        "origin_xyz_A": [0.0, 0.0, 0.0],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Graph topology + radii -> binary MRC mask.")
    parser.add_argument("--i", required=True, help="Input config YAML path (e.g. config.yaml).")
    return parser


def _cmd_build_mask(cfg: Mapping[str, Any]) -> int:
    task = str(cfg["task"])
    topology = load_graph_topology(str(_require(cfg, "topology", task)))
    node_radii = load_node_radii(str(_require(cfg, "node_radii", task)))
    edge_radii = load_edge_radii(str(_require(cfg, "edge_radii", task)))

    grid_raw = _require(cfg, "grid_config", task)
    if isinstance(grid_raw, Mapping):
        grid_config = normalize_grid_config(grid_raw)
    else:
        with open(str(grid_raw), "r", encoding="utf-8") as f:
            grid_config = normalize_grid_config(json.load(f))

    output_mrc = str(_require(cfg, "output_mrc", task))

    validated = validate_graph_inputs(topology, node_radii, edge_radii)
    warnings = validated["warnings"]
    if warnings["extra_node_radii"] or warnings["extra_edge_radii"]:
        print(
            json.dumps(
                {"warnings": warnings},
                indent=2,
                sort_keys=True,
            )
        )

    mask = graph_to_mask(grid_config, topology, node_radii, edge_radii, dtype=np.uint8)
    mask_to_mrc(
        output_path=output_mrc,
        mask=mask,
        voxel_size_xyz_A=grid_config["voxel_size_xyz_A"],
        origin_xyz_A=grid_config["origin_xyz_A"],
        overwrite=True,
    )
    print(
        json.dumps(
            {
                "output_mrc": output_mrc,
                "mask_shape_zyx": list(mask.shape),
                "unique_values": [int(v) for v in np.unique(mask)],
                "occupied_voxels": int(np.count_nonzero(mask)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _cmd_example_inputs(cfg: Mapping[str, Any]) -> int:
    task = str(cfg["task"])
    outdir = Path(str(_require(cfg, "outdir", task)))
    outdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "graph_topology.json": _template_topology(),
        "node_radii.json": _template_node_radii(),
        "edge_radii.json": _template_edge_radii(),
        "grid_config.json": _template_grid_config(),
    }
    for name, payload in outputs.items():
        with (outdir / name).open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
    print(json.dumps({"written_files": sorted(outputs.keys()), "outdir": str(outdir)}, indent=2))
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = _load_yaml_config(args.i)
    task = str(_require(cfg, "task", "root"))
    if task == "build-mask":
        return _cmd_build_mask(cfg)
    if task == "example-inputs":
        return _cmd_example_inputs(cfg)
    raise ValueError("Unknown task. Expected one of: build-mask, example-inputs.")


if __name__ == "__main__":
    raise SystemExit(main())
