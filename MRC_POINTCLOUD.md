# MRC PointCloud Module Documentation

## Overview

`mrc_pointcloud.py` provides bidirectional conversion between:

- MRC density volume (`.mrc`)
- Chroma-compatible point cloud (`numpy.ndarray` with shape `(N, 3)`)

It is designed for the `ShapeConditioner` workflow in `ChromaAPI.ipynb`.

Core goals:

- Read MRC data and metadata (`voxel_size`, `origin`)
- Inspect density statistics quickly to choose threshold
- Convert MRC -> point cloud using threshold selection
- Convert point cloud -> MRC using occupancy/count/gaussian rasterization

---

## Requirements

- Python 3.10+
- `numpy`
- `mrcfile`
- `PyYAML`

You already installed `mrcfile` in your conda environment.

---

## Coordinate Convention

This module uses **physical MRC coordinates**:

- Voxel index order in arrays is `(z, y, x)`
- Point cloud coordinate order is `(x, y, z)`
- Mapping:
  - `x = origin_x + i_x * voxel_size_x`
  - `y = origin_y + i_y * voxel_size_y`
  - `z = origin_z + i_z * voxel_size_z`

This is aligned with your requirement for physical-space conversion.

---

## Python API

### 1) `read_mrc(path)`

Read MRC file and return:

- `data`: `float32` array of shape `(Z, Y, X)`
- `voxel_size`: tuple `(x, y, z)`
- `origin`: tuple `(x, y, z)`
- `header_info`: basic metadata dictionary

```python
import mrc_pointcloud as mpc

data, voxel_size, origin, header = mpc.read_mrc("input.mrc")
```

### 2) `inspect_mrc(data, quantiles=(0.5, 0.9, 0.95, 0.99))`

Compute fast statistics for threshold selection:

- min, max, mean, std, median
- nonzero fraction
- quantiles (e.g. `q90`, `q95`, `q99`)

```python
stats = mpc.inspect_mrc(data, quantiles=(0.5, 0.9, 0.95, 0.99))
print(stats["q95"])
```

### 3) `mrc_to_point_cloud(...)`

Convert density volume to point cloud with threshold:

- Select voxels where `density >= threshold`
- Optionally downsample to `max_points`
- Output shape is `(N, 3)`

```python
points = mpc.mrc_to_point_cloud(
    data,
    voxel_size=voxel_size,
    origin=origin,
    threshold=stats["q95"],
    max_points=20000,
    random_seed=0,
)
print(points.shape)  # (N, 3)
```

### 4) `point_cloud_to_mrc(...)`

Rasterize point cloud back to MRC grid.

Supported modes:

- `occupancy`: hit voxel = 1
- `count`: count hits per voxel
- `gaussian`: gaussian deposition around points (`sigma` required)

```python
recon = mpc.point_cloud_to_mrc(
    points,
    voxel_size=voxel_size,
    origin=origin,
    grid_shape=data.shape,
    mode="occupancy",
)
```

### 5) `write_mrc(path, data, voxel_size, origin, overwrite=True)`

Write `(Z, Y, X)` density grid to `.mrc`.

```python
mpc.write_mrc("recon.mrc", recon, voxel_size=voxel_size, origin=origin)
```

---

## CLI Usage

All CLI tasks are config-driven:

```bash
python mrc_pointcloud.py --i config.yaml
```

### Inspect MRC statistics

```bash
python mrc_pointcloud.py --i ./configs/mrc_inspect.yaml
```

### Convert MRC -> point cloud (`.npy`)

```bash
python mrc_pointcloud.py --i ./configs/mrc_to_pc.yaml
```

To enforce strict threshold selection with no random downsampling, leave
`max_points` and `seed` unset in YAML.

### Convert point cloud (`.npy`) -> MRC

```bash
python mrc_pointcloud.py --i ./configs/mrc_to_mrc.yaml
```

Gaussian mode example:

```bash
python mrc_pointcloud.py --i ./configs/mrc_to_mrc.yaml
```

Set `mode: gaussian` and `sigma: <value>` inside that YAML file.

---

## Integration with Chroma ShapeConditioner

```python
import mrc_pointcloud as mpc
from chroma.layers.structure import conditioners

data, voxel_size, origin, _ = mpc.read_mrc("target_shape.mrc")
stats = mpc.inspect_mrc(data)

letter_point_cloud = mpc.mrc_to_point_cloud(
    data,
    voxel_size=voxel_size,
    origin=origin,
    threshold=stats["q95"],  # tune as needed
    max_points=2000,         # tune as needed
    random_seed=0,
)

conditioner = conditioners.ShapeConditioner(
    letter_point_cloud,
    chroma.backbone_network.noise_schedule,
    autoscale_num_residues=NUM_RESIDUES
).to(device)
```

---

## Threshold Tuning Tips

- Start from quantiles:
  - conservative shape: try `q95` or `q99`
  - fuller shape: try `q90` or lower
- If point cloud is too dense:
  - increase threshold
  - or set `max_points`
- If point cloud is too sparse:
  - lower threshold

---

## Validation Checklist

For each new MRC:

1. `inspect` to view value range and quantiles
2. run `to-pc` with threshold
3. verify point cloud shape `(N, 3)`
4. run `to-mrc` for reconstruction check
5. inspect reconstructed MRC if needed

---

## Error Handling Notes

The module raises `ValueError` for common invalid inputs:

- wrong array dimensions
- non-finite values
- invalid `voxel_size` / `origin`
- empty point cloud
- unsupported rasterization mode
- no voxel selected at current threshold

---

## File Location

- Module: `mrc_pointcloud.py`
- Documentation: `MRC_POINTCLOUD.md`

