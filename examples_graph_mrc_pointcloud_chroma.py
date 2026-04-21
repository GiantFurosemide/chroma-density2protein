"""End-to-end example: Graph -> MRC -> PointCloud -> Chroma-ready payload.

This script creates toy graph inputs, builds a binary mask MRC, converts it to
a point cloud, and stores outputs for downstream Chroma usage.

Optional: if Chroma is installed and initialized in your environment, you can
enable the example sampling block near the end of this file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

import graph_to_mrc as g2m
import mrc_pointcloud as mpc


def make_toy_inputs() -> Dict[str, dict]:
    topology = {
        "nodes": [
            {"id": "N1", "coord": [20.0, 30.0, 20.0]},
            {"id": "N2", "coord": [45.0, 30.0, 20.0]},
            {"id": "N3", "coord": [32.5, 48.0, 20.0]},
        ],
        "edges": [
            {"id": "E1", "source": "N1", "target": "N2"},
            {"id": "E2", "source": "N2", "target": "N3"},
            {"id": "E3", "source": "N3", "target": "N1"},
        ],
    }
    node_radii = {"N1": 4.0, "N2": 5.0, "N3": 4.5}
    edge_radii = {"E1": 2.0, "E2": 1.8, "E3": 1.8}
    grid_config = {
        "box_shape_zyx": [96, 96, 96],
        "voxel_size_xyz_A": [1.0, 1.0, 1.0],
        "origin_xyz_A": [0.0, 0.0, 0.0],
    }
    return {
        "topology": topology,
        "node_radii": node_radii,
        "edge_radii": edge_radii,
        "grid_config": grid_config,
    }


def save_inputs(base_dir: Path, payload: Dict[str, dict]) -> Dict[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "topology": base_dir / "graph_topology.json",
        "node_radii": base_dir / "node_radii.json",
        "edge_radii": base_dir / "edge_radii.json",
        "grid_config": base_dir / "grid_config.json",
    }
    for key, path in paths.items():
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload[key], f, indent=2, sort_keys=True)
            f.write("\n")
    return paths


def main() -> None:
    root = Path(".")
    input_dir = root / "example_graph_inputs"
    output_dir = root / "example_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = make_toy_inputs()
    input_paths = save_inputs(input_dir, payload)

    topology = g2m.load_graph_topology(str(input_paths["topology"]))
    node_radii = g2m.load_node_radii(str(input_paths["node_radii"]))
    edge_radii = g2m.load_edge_radii(str(input_paths["edge_radii"]))
    grid_config = g2m.normalize_grid_config(payload["grid_config"])

    mask = g2m.graph_to_mask(
        grid_config=grid_config,
        topology=topology,
        node_radii=node_radii,
        edge_radii=edge_radii,
        dtype=np.uint8,
    )
    unique_vals = set(np.unique(mask).tolist())
    if not unique_vals.issubset({0, 1}):
        raise RuntimeError(f"Mask is not binary. Unique values: {sorted(unique_vals)}")

    mask_mrc_path = output_dir / "graph_mask.mrc"
    g2m.mask_to_mrc(
        output_path=str(mask_mrc_path),
        mask=mask,
        voxel_size_xyz_A=grid_config["voxel_size_xyz_A"],
        origin_xyz_A=grid_config["origin_xyz_A"],
        overwrite=True,
    )

    data, voxel_size, origin, _ = mpc.read_mrc(str(mask_mrc_path))
    stats = mpc.inspect_mrc(data)
    # Binary mask convention: threshold 0.5 to keep foreground voxels.
    points = mpc.mrc_to_point_cloud(
        data,
        voxel_size=voxel_size,
        origin=origin,
        threshold=0.5,
        max_points=10000,
        random_seed=0,
    )

    points_npy_path = output_dir / "graph_points.npy"
    np.save(points_npy_path, points)

    points_meta_path = output_dir / "graph_points_for_chroma.npz"
    np.savez(
        points_meta_path,
        points=points,
        voxel_size=np.asarray(voxel_size, dtype=np.float32),
        origin=np.asarray(origin, dtype=np.float32),
    )

    print(json.dumps(
        {
            "inputs_dir": str(input_dir),
            "outputs_dir": str(output_dir),
            "mask_mrc": str(mask_mrc_path),
            "points_npy": str(points_npy_path),
            "points_npz": str(points_meta_path),
            "mask_shape_zyx": list(mask.shape),
            "point_cloud_shape": list(points.shape),
            "mrc_stats_min_max": [stats["min"], stats["max"]],
            "threshold_used": 0.5,
        },
        indent=2,
        sort_keys=True,
    ))

    # Optional Chroma sketch (enable in your Chroma runtime environment):
    #
    # from chroma.layers.structure import conditioners
    # import torch
    #
    # NUM_RESIDUES = 800
    # conditioner = conditioners.ShapeConditioner(
    #     points,
    #     chroma.backbone_network.noise_schedule,
    #     autoscale_num_residues=NUM_RESIDUES
    # ).to(device)
    # torch.manual_seed(0)
    # protein = chroma.sample(chain_lengths=[NUM_RESIDUES], conditioner=conditioner)
    # protein.to(str(output_dir / "graph_conditioned_sample.pdb"))


if __name__ == "__main__":
    main()
