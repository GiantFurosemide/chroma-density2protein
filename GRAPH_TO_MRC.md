# Graph to MRC Module Documentation

## Overview

`graph_to_mrc.py` converts a graph-defined shape into a binary MRC mask.

The shape is defined as:

- Union of node spheres
- Union of edge cylinders
- Final mask is the union of both spaces

Mask convention:

- voxel value `1`: inside shape
- voxel value `0`: background

---

## Design Principles

This module follows your decoupled design:

1. Topology and radii are separated.
2. Physical coordinate system is used consistently.
3. Units are Angstrom for geometry parameters.

All geometric computations are done in physical coordinate space (x, y, z in Angstrom), not in voxel index space.

---

## Input Files (Decoupled)

### 1) `graph_topology.json`

Topology only (no radii):

```json
{
  "nodes": [
    { "id": "N1", "coord": [20.0, 30.0, 25.0] },
    { "id": "N2", "coord": [40.0, 30.0, 25.0] }
  ],
  "edges": [
    { "id": "E1", "source": "N1", "target": "N2" }
  ]
}
```

### 2) `node_radii.json`

Node radius map:

```json
{
  "N1": 4.0,
  "N2": 5.0
}
```

### 3) `edge_radii.json`

Edge radius map:

```json
{
  "E1": 2.0
}
```

### 4) `grid_config.json`

Grid and MRC metadata:

```json
{
  "box_shape_zyx": [80, 80, 80],
  "voxel_size_xyz_A": [1.0, 1.0, 1.0],
  "origin_xyz_A": [0.0, 0.0, 0.0]
}
```

---

## Coordinate and Grid Convention

- Node coordinates are `[x, y, z]` in Angstrom.
- Grid shape is `[Z, Y, X]` in voxel count.
- Voxel center coordinates:
  - `x = origin_x + ix * voxel_size_x`
  - `y = origin_y + iy * voxel_size_y`
  - `z = origin_z + iz * voxel_size_z`

---

## Geometry Definition

### Node Sphere

A voxel is inside node `i` if:

`||p - center_i|| <= radius_node_i`

### Edge Cylinder

An edge cylinder is defined by source and target node centers.

For voxel center `p`, compute nearest point on line segment `(source -> target)`.
Voxel is inside edge `e` if nearest distance to the segment is:

`distance(p, segment_e) <= radius_edge_e`

Degenerate edges (same source/target position) are treated as spheres with edge radius.

---

## Python API

### Loaders

- `load_graph_topology(path_or_dict)`
- `load_node_radii(path_or_dict)`
- `load_edge_radii(path_or_dict)`
- `normalize_grid_config(grid_config)`

### Validation

- `validate_graph_inputs(topology, node_radii, edge_radii)`

### Build mask

- `graph_to_mask(grid_config, topology, node_radii, edge_radii, dtype=np.uint8)`

Returns `np.ndarray` with shape `(Z, Y, X)` and values `{0, 1}`.

### Write mask as MRC

- `mask_to_mrc(output_path, mask, voxel_size_xyz_A, origin_xyz_A, overwrite=True)`

---

## CLI

All CLI tasks are config-driven:

```bash
python graph_to_mrc.py --i config.yaml
```

## 1) Generate template inputs

```bash
python graph_to_mrc.py --i ./configs/graph_example_inputs.yaml
```

Outputs:

- `graph_inputs/graph_topology.json`
- `graph_inputs/node_radii.json`
- `graph_inputs/edge_radii.json`
- `graph_inputs/grid_config.json`

## 2) Build MRC mask

```bash
python graph_to_mrc.py --i ./configs/graph_build_mask.yaml
```

---

## Industrial-Style Conventions

Recommended conventions:

1. Keep IDs stable and explicit (`N1`, `N2`, `E1`, etc.).
2. Keep topology immutable and versionable.
3. Keep geometry/radius parameter files configurable by scenario.
4. Keep grid configuration in runtime config (not hardcoded).

This separation makes it easier to:

- change radii without changing topology
- keep multiple design variants
- integrate with workflow engines and parameter sweeps

---

## Validation Rules Implemented

The module enforces:

- node/edge IDs are unique
- edge source/target refer to existing node IDs
- node radii must exist for every topology node
- edge radii must exist for every topology edge
- all radii are finite and non-negative
- box shape has 3 positive integers
- voxel size has 3 positive floats
- origin has 3 finite floats

It also reports extra IDs in radii files as warnings.

---

## Troubleshooting

### No occupied voxels

Possible causes:

- all radii too small relative to voxel size
- objects outside configured box extents

Actions:

- increase box size
- move origin
- reduce voxel size
- increase radii

### Shape appears clipped

Cause:

- box does not cover whole physical geometry

Action:

- expand `box_shape_zyx` and/or adjust `origin_xyz_A`

### Performance is slow for very large boxes

Cause:

- 3D voxelization scales with volume size

Action:

- choose tighter box extents
- use coarser voxel size for early exploration

---

## Minimal Python Example

```python
import graph_to_mrc as g2m

topology = g2m.load_graph_topology("graph_topology.json")
node_radii = g2m.load_node_radii("node_radii.json")
edge_radii = g2m.load_edge_radii("edge_radii.json")

grid = g2m.normalize_grid_config({
    "box_shape_zyx": [80, 80, 80],
    "voxel_size_xyz_A": [1.0, 1.0, 1.0],
    "origin_xyz_A": [0.0, 0.0, 0.0],
})

mask = g2m.graph_to_mask(grid, topology, node_radii, edge_radii)
g2m.mask_to_mrc(
    "graph_mask.mrc",
    mask,
    voxel_size_xyz_A=grid["voxel_size_xyz_A"],
    origin_xyz_A=grid["origin_xyz_A"],
)
```

