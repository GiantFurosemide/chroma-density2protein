"""MRC <-> point cloud conversion utilities for Chroma ShapeConditioner.

This module provides a small API and CLI for converting density maps in MRC
format into Chroma-compatible point clouds (N, 3), and rasterizing point
clouds back into MRC grids.

Example (Python):
    import mrc_pointcloud as mpc

    data, voxel_size, origin, _ = mpc.read_mrc("input.mrc")
    stats = mpc.inspect_mrc(data)
    points = mpc.mrc_to_point_cloud(
        data, voxel_size=voxel_size, origin=origin, threshold=stats["q95"]
    )
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mrcfile
import numpy as np
import yaml


Array3D = np.ndarray
PointCloud = np.ndarray
Vec3 = Tuple[float, float, float]
Shape3 = Tuple[int, int, int]


def _ensure_3d_array(data: np.ndarray, name: str = "data") -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3D array with shape (Z, Y, X).")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _ensure_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if arr.shape[0] == 0:
        raise ValueError("points must not be empty.")
    if not np.isfinite(arr).all():
        raise ValueError("points contain non-finite values.")
    return arr


def _normalize_voxel_size(voxel_size: Sequence[float] | float) -> Vec3:
    if np.isscalar(voxel_size):
        value = float(voxel_size)
        out = np.array([value, value, value], dtype=np.float64)
    else:
        out = np.asarray(voxel_size, dtype=np.float64).reshape(-1)
        if out.size == 1:
            out = np.repeat(out.item(), 3)
        elif out.size != 3:
            raise ValueError("voxel_size must be a float or a 3-value sequence (x, y, z).")
    if np.any(out <= 0):
        raise ValueError("voxel_size values must be positive.")
    return float(out[0]), float(out[1]), float(out[2])


def _normalize_origin(origin: Sequence[float] | None) -> Vec3:
    if origin is None:
        return 0.0, 0.0, 0.0
    arr = np.asarray(origin, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError("origin must be a 3-value sequence (x, y, z).")
    if not np.isfinite(arr).all():
        raise ValueError("origin contains non-finite values.")
    return float(arr[0]), float(arr[1]), float(arr[2])


def _parse_quantiles(values: Sequence[float]) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64).reshape(-1)
    if q.size == 0:
        raise ValueError("quantiles must not be empty.")
    if np.any((q < 0.0) | (q > 1.0)):
        raise ValueError("quantiles must be in [0, 1].")
    return q


def read_mrc(path: str) -> Tuple[Array3D, Vec3, Vec3, Dict[str, object]]:
    """Read an MRC file and return data, voxel_size, origin, and header info.

    Returns:
        data: Float32 array with shape (Z, Y, X).
        voxel_size: Physical voxel size as (x, y, z).
        origin: Physical origin as (x, y, z).
        header_info: Basic metadata dictionary.
    """
    with mrcfile.open(path, mode="r", permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float32).copy()
        vx = float(mrc.voxel_size.x)
        vy = float(mrc.voxel_size.y)
        vz = float(mrc.voxel_size.z)
        voxel_size = _normalize_voxel_size((vx, vy, vz))
        origin = _normalize_origin(
            (float(mrc.header.origin.x), float(mrc.header.origin.y), float(mrc.header.origin.z))
        )

    header_info: Dict[str, object] = {
        "shape_zyx": tuple(int(v) for v in data.shape),
        "dtype": str(data.dtype),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
    }
    return data, voxel_size, origin, header_info


def inspect_mrc(data: np.ndarray, quantiles: Sequence[float] = (0.5, 0.9, 0.95, 0.99)) -> Dict[str, float]:
    """Compute fast density statistics for threshold selection."""
    arr = _ensure_3d_array(np.asarray(data), name="data").astype(np.float64, copy=False)
    q = _parse_quantiles(quantiles)
    q_values = np.quantile(arr, q=q)

    stats: Dict[str, float] = {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "nonzero_fraction": float(np.count_nonzero(arr) / arr.size),
    }
    for qi, val in zip(q, q_values):
        key = f"q{int(round(qi * 100))}"
        stats[key] = float(val)
    return stats


def mrc_to_point_cloud(
    data: np.ndarray,
    voxel_size: Sequence[float] | float,
    origin: Sequence[float] | None,
    threshold: float,
    max_points: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> PointCloud:
    """Convert MRC density to physical XYZ point cloud by thresholding.

    Args:
        data: Density array with shape (Z, Y, X).
        voxel_size: Voxel spacing as scalar or (x, y, z).
        origin: Physical coordinate origin (x, y, z).
        threshold: Keep voxels where density >= threshold.
        max_points: Optional cap. Uniform random downsampling if exceeded.
        random_seed: Optional deterministic seed for downsampling.
    """
    arr = _ensure_3d_array(np.asarray(data), name="data")
    vx, vy, vz = _normalize_voxel_size(voxel_size)
    ox, oy, oz = _normalize_origin(origin)

    mask = arr >= float(threshold)
    idx_zyx = np.argwhere(mask)
    if idx_zyx.size == 0:
        raise ValueError("No voxels selected. Try lowering threshold.")

    if max_points is not None:
        if max_points <= 0:
            raise ValueError("max_points must be positive when provided.")
        if idx_zyx.shape[0] > max_points:
            rng = np.random.default_rng(seed=random_seed)
            keep = rng.choice(idx_zyx.shape[0], size=max_points, replace=False)
            idx_zyx = idx_zyx[keep]

    z = idx_zyx[:, 0].astype(np.float64)
    y = idx_zyx[:, 1].astype(np.float64)
    x = idx_zyx[:, 2].astype(np.float64)
    points = np.stack(
        [ox + x * vx, oy + y * vy, oz + z * vz],
        axis=1,
    ).astype(np.float32)
    return points


def fft_lowpass_spherical_rfft(
    data: np.ndarray,
    cutoff_cycles_per_pixel: float = 0.25,
) -> np.ndarray:
    """Spherical low-pass in Fourier space (zero high |k|), anti-aliasing before 2-fold binning.

    Uses normalized fftfreq coordinates where Nyquist is 0.5; default cutoff 0.25 keeps
    frequencies safe for factor-2 decimation (half Nyquist of the fine grid).

    Args:
        data: Real 3D array (Z, Y, X).
        cutoff_cycles_per_pixel: Euclidean radius in frequency space (same units as fftfreq).

    Returns:
        Real-space filtered array, float32, same shape as input.
    """
    arr = _ensure_3d_array(np.asarray(data), name="data").astype(np.float64, copy=False)
    nz, ny, nx = arr.shape
    spec = np.fft.rfftn(arr)
    kz = np.fft.fftfreq(nz)[:, None, None]
    ky = np.fft.fftfreq(ny)[None, :, None]
    kx = np.fft.rfftfreq(nx)[None, None, :]
    k2 = kz * kz + ky * ky + kx * kx
    r_cut = float(cutoff_cycles_per_pixel)
    mask = k2 <= r_cut * r_cut
    spec *= mask
    out = np.fft.irfftn(spec, s=(nz, ny, nx))
    return np.asarray(out, dtype=np.float32)


def bin2_mean_volume(data: np.ndarray) -> np.ndarray:
    """Average-pool 2×2×2 blocks (mean). Odd dimensions: crop trailing slice per axis."""
    arr = _ensure_3d_array(np.asarray(data), name="data")
    z, y, x = arr.shape
    if z < 2 or y < 2 or x < 2:
        raise ValueError("bin2_mean_volume requires each dimension to be at least 2.")
    z2, y2, x2 = z - (z % 2), y - (y % 2), x - (x % 2)
    a = arr[:z2, :y2, :x2]
    return (
        a.reshape(z2 // 2, 2, y2 // 2, 2, x2 // 2, 2)
        .mean(axis=(1, 3, 5))
        .astype(np.float32)
    )


def bin2_max_volume(data: np.ndarray) -> np.ndarray:
    """Max-pool 2×2×2 blocks. Odd dimensions: crop trailing slice per axis (same as mean bin)."""
    arr = _ensure_3d_array(np.asarray(data), name="data")
    z, y, x = arr.shape
    if z < 2 or y < 2 or x < 2:
        raise ValueError("bin2_max_volume requires each dimension to be at least 2.")
    z2, y2, x2 = z - (z % 2), y - (y % 2), x - (x % 2)
    a = arr[:z2, :y2, :x2]
    return (
        a.reshape(z2 // 2, 2, y2 // 2, 2, x2 // 2, 2)
        .max(axis=(1, 3, 5))
        .astype(np.float32)
    )


def mrc_to_point_cloud_adaptive_cap(
    data: np.ndarray,
    voxel_size: Sequence[float] | float,
    origin: Sequence[float] | None,
    threshold: float,
    max_voxels: int = 2000,
    max_rounds: int = 32,
    subsample_seed: Optional[int] = 0,
    cutoff_cycles_per_pixel: float = 0.25,
    adaptive_pooling: str = "mean_fft",
) -> Tuple[PointCloud, Dict[str, Any], Array3D]:
    """Threshold-select voxels; if too many, coarsen the grid until count <= max_voxels.

    Physical voxel_size is doubled each bin round; origin is unchanged. Point coordinates use
    the same corner-based mapping as ``mrc_to_point_cloud`` on the **final** grid.

    If after ``max_rounds`` the count still exceeds ``max_voxels``, falls back to uniform random
    subsampling of selected voxel centers (deterministic if ``subsample_seed`` is set).

    ``adaptive_pooling``:
        - ``"mean_fft"`` (default): FFT low-pass anti-aliasing, then 2×2×2 **mean** pooling.
          Use ``threshold`` to select foreground on possibly non-binary data.
        - ``"binary_max"``: no FFT; 2×2×2 **max** pooling so each coarse voxel is 1 iff any child
          was foreground. Keeps strict 0/1 semantics after binarization at ``threshold``; counting
          uses nonzero voxels (equivalent to ``>= 0.5`` for values in ``{0, 1}``).

    If ``data`` is a binarized mask and you use ``mean_fft``, pass ``threshold=0.5`` so foreground
    stays interpretable after smoothing and mean pooling.

    Returns:
        points, meta, final_density_zyx — the last array is the processed (Z,Y,X) grid used for extraction.
    """
    if max_voxels <= 0:
        raise ValueError("max_voxels must be positive.")
    mode = str(adaptive_pooling).strip().lower()
    if mode not in ("mean_fft", "binary_max"):
        raise ValueError('adaptive_pooling must be "mean_fft" or "binary_max".')

    arr = _ensure_3d_array(np.asarray(data), name="data").astype(np.float32, copy=True)
    vx, vy, vz = _normalize_voxel_size(voxel_size)
    ox, oy, oz = _normalize_origin(origin)
    thr = float(threshold)

    if mode == "binary_max":
        arr = (arr >= thr).astype(np.float32)

    def _foreground_count(a: np.ndarray) -> int:
        if mode == "binary_max":
            return int(np.count_nonzero(a))
        return int(np.count_nonzero(a >= thr))

    history: List[Dict[str, Any]] = []
    rounds = 0
    initial_count = _foreground_count(arr)

    while True:
        count = _foreground_count(arr)
        history.append(
            {
                "round": rounds,
                "shape_zyx": [int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])],
                "voxel_size_xyz": [vx, vy, vz],
                "selected_voxels_ge_threshold": count,
            }
        )
        if count == 0:
            raise ValueError("No voxels >= threshold after preprocessing. Lower threshold.")
        if count <= max_voxels:
            break
        if rounds >= max_rounds:
            break
        z, y, x = arr.shape
        if z < 2 or y < 2 or x < 2:
            break

        if mode == "mean_fft":
            arr = fft_lowpass_spherical_rfft(arr, cutoff_cycles_per_pixel=cutoff_cycles_per_pixel)
            arr = bin2_mean_volume(arr)
        else:
            arr = bin2_max_volume(arr)
        vx, vy, vz = vx * 2.0, vy * 2.0, vz * 2.0
        rounds += 1

    extract_thr = 0.5 if mode == "binary_max" else thr
    count = _foreground_count(arr)
    selected_count_le_max_voxels_before_subsample = count <= max_voxels
    used_subsample = False
    if count > max_voxels:
        points = mrc_to_point_cloud(
            arr,
            voxel_size=(vx, vy, vz),
            origin=(ox, oy, oz),
            threshold=extract_thr,
            max_points=max_voxels,
            random_seed=subsample_seed,
        )
        used_subsample = True
    else:
        points = mrc_to_point_cloud(
            arr,
            voxel_size=(vx, vy, vz),
            origin=(ox, oy, oz),
            threshold=extract_thr,
            max_points=None,
            random_seed=None,
        )

    meta: Dict[str, Any] = {
        "threshold": thr,
        "adaptive_pooling": mode,
        "extraction_threshold": extract_thr,
        "max_voxels": max_voxels,
        "max_rounds": max_rounds,
        "initial_selected_voxels_ge_threshold": initial_count,
        "final_shape_zyx": [int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])],
        "final_voxel_size_xyz": [vx, vy, vz],
        "bin_rounds_applied": rounds,
        "history": history,
        "final_selected_voxels_ge_threshold_before_subsample": count,
        "selected_count_le_max_voxels_before_subsample": selected_count_le_max_voxels_before_subsample,
        "output_points": int(points.shape[0]),
        "used_random_subsample": used_subsample,
        "subsample_seed": subsample_seed,
    }
    return points, meta, arr


def _infer_origin_and_shape(points: np.ndarray, voxel_size: Vec3) -> Tuple[Vec3, Shape3]:
    vx, vy, vz = voxel_size
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)

    origin = (float(mins[0]), float(mins[1]), float(mins[2]))
    nx = int(np.floor((maxs[0] - origin[0]) / vx)) + 1
    ny = int(np.floor((maxs[1] - origin[1]) / vy)) + 1
    nz = int(np.floor((maxs[2] - origin[2]) / vz)) + 1
    return origin, (nz, ny, nx)


def point_cloud_to_mrc(
    points: np.ndarray,
    voxel_size: Sequence[float] | float,
    origin: Sequence[float] | None = None,
    grid_shape: Optional[Sequence[int]] = None,
    mode: str = "occupancy",
    sigma: Optional[float] = None,
) -> Array3D:
    """Rasterize physical XYZ point cloud into (Z, Y, X) MRC grid.

    Modes:
        occupancy: set hit voxels to 1.
        count: increment voxel hit counts.
        gaussian: gaussian deposition around each point (requires sigma > 0).
    """
    pts = _ensure_points(points)
    vx, vy, vz = _normalize_voxel_size(voxel_size)

    if grid_shape is None:
        if origin is None:
            origin_xyz, shape_zyx = _infer_origin_and_shape(pts, (vx, vy, vz))
        else:
            origin_xyz = _normalize_origin(origin)
            rel = (pts - np.array(origin_xyz, dtype=np.float64)) / np.array([vx, vy, vz], dtype=np.float64)
            max_idx = np.max(np.floor(rel), axis=0)
            if np.any(max_idx < 0):
                raise ValueError("Provided origin places all points outside the grid.")
            shape_zyx = (
                int(max_idx[2]) + 1,
                int(max_idx[1]) + 1,
                int(max_idx[0]) + 1,
            )
    else:
        shp = tuple(int(v) for v in grid_shape)
        if len(shp) != 3 or any(v <= 0 for v in shp):
            raise ValueError("grid_shape must be 3 positive integers as (Z, Y, X).")
        shape_zyx = shp
        origin_xyz = _normalize_origin(origin)

    grid = np.zeros(shape_zyx, dtype=np.float32)
    rel = (pts - np.array(origin_xyz, dtype=np.float64)) / np.array([vx, vy, vz], dtype=np.float64)

    if mode not in {"occupancy", "count", "gaussian"}:
        raise ValueError("mode must be one of: occupancy, count, gaussian.")

    if mode in {"occupancy", "count"}:
        idx_xyz = np.rint(rel).astype(np.int64)
        ix = idx_xyz[:, 0]
        iy = idx_xyz[:, 1]
        iz = idx_xyz[:, 2]
        in_bounds = (
            (ix >= 0)
            & (ix < shape_zyx[2])
            & (iy >= 0)
            & (iy < shape_zyx[1])
            & (iz >= 0)
            & (iz < shape_zyx[0])
        )
        ix = ix[in_bounds]
        iy = iy[in_bounds]
        iz = iz[in_bounds]

        if mode == "occupancy":
            grid[iz, iy, ix] = 1.0
        else:
            np.add.at(grid, (iz, iy, ix), 1.0)
        return grid

    if sigma is None or sigma <= 0:
        raise ValueError("sigma must be provided and positive for gaussian mode.")

    sigma_vox = np.array([sigma / vx, sigma / vy, sigma / vz], dtype=np.float64)
    radius = np.maximum(np.ceil(3.0 * sigma_vox).astype(np.int64), 1)

    for p in rel:
        cx, cy, cz = p
        x0 = max(int(np.floor(cx - radius[0])), 0)
        x1 = min(int(np.ceil(cx + radius[0])) + 1, shape_zyx[2])
        y0 = max(int(np.floor(cy - radius[1])), 0)
        y1 = min(int(np.ceil(cy + radius[1])) + 1, shape_zyx[1])
        z0 = max(int(np.floor(cz - radius[2])), 0)
        z1 = min(int(np.ceil(cz + radius[2])) + 1, shape_zyx[0])

        if x0 >= x1 or y0 >= y1 or z0 >= z1:
            continue

        zz, yy, xx = np.meshgrid(
            np.arange(z0, z1, dtype=np.float64),
            np.arange(y0, y1, dtype=np.float64),
            np.arange(x0, x1, dtype=np.float64),
            indexing="ij",
        )
        dx = (xx - cx) / sigma_vox[0]
        dy = (yy - cy) / sigma_vox[1]
        dz = (zz - cz) / sigma_vox[2]
        kernel = np.exp(-0.5 * (dx * dx + dy * dy + dz * dz))
        grid[z0:z1, y0:y1, x0:x1] += kernel.astype(np.float32)

    return grid


def write_mrc(
    path: str,
    data: np.ndarray,
    voxel_size: Sequence[float] | float,
    origin: Sequence[float] | None = None,
    overwrite: bool = True,
) -> None:
    """Write a (Z, Y, X) density grid to MRC file."""
    arr = _ensure_3d_array(np.asarray(data), name="data").astype(np.float32, copy=False)
    vx, vy, vz = _normalize_voxel_size(voxel_size)
    ox, oy, oz = _normalize_origin(origin)

    with mrcfile.new(path, overwrite=overwrite) as mrc:
        mrc.set_data(arr)
        mrc.voxel_size = (vx, vy, vz)
        mrc.header.origin.x = ox
        mrc.header.origin.y = oy
        mrc.header.origin.z = oz
        mrc.update_header_from_data()
        mrc.update_header_stats()


def _load_config(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError("Config must be a YAML object at top level.")
    return loaded


def _require(cfg: Dict[str, object], key: str, task: str) -> object:
    if key not in cfg:
        raise ValueError(f"Missing required key '{key}' for task '{task}'.")
    return cfg[key]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MRC <-> point cloud conversion tools.")
    parser.add_argument("--i", required=True, help="Input config YAML path (e.g. config.yaml).")
    return parser


def _cli_inspect(cfg: Dict[str, object]) -> int:
    task = str(cfg["task"])
    input_mrc = str(_require(cfg, "input_mrc", task))
    quantiles = cfg.get("quantiles", [0.5, 0.9, 0.95, 0.99])
    data, voxel_size, origin, header_info = read_mrc(input_mrc)
    stats = inspect_mrc(data, quantiles=quantiles)
    payload = {"voxel_size": voxel_size, "origin": origin, "header": header_info, "stats": stats}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cli_to_pc(cfg: Dict[str, object]) -> int:
    task = str(cfg["task"])
    input_mrc = str(_require(cfg, "input_mrc", task))
    output_npy = str(_require(cfg, "output_npy", task))
    threshold = float(_require(cfg, "threshold", task))
    max_points = cfg.get("max_points")
    seed = cfg.get("seed")
    data, voxel_size, origin, _ = read_mrc(input_mrc)
    points = mrc_to_point_cloud(
        data,
        voxel_size=voxel_size,
        origin=origin,
        threshold=threshold,
        max_points=None if max_points is None else int(max_points),
        random_seed=None if seed is None else int(seed),
    )
    np.save(output_npy, points)
    print(f"Saved point cloud: {output_npy}, shape={points.shape}")
    return 0


def _cli_to_mrc(cfg: Dict[str, object]) -> int:
    task = str(cfg["task"])
    input_npy = str(_require(cfg, "input_npy", task))
    output_mrc = str(_require(cfg, "output_mrc", task))
    voxel_size = _normalize_voxel_size(_require(cfg, "voxel_size", task))
    origin = cfg.get("origin")
    grid_shape = cfg.get("grid_shape")
    mode = str(cfg.get("mode", "occupancy"))
    sigma = cfg.get("sigma")
    points = np.load(input_npy)
    grid = point_cloud_to_mrc(
        points=points,
        voxel_size=voxel_size,
        origin=origin,
        grid_shape=grid_shape,
        mode=mode,
        sigma=None if sigma is None else float(sigma),
    )
    write_mrc(
        path=output_mrc,
        data=grid,
        voxel_size=voxel_size,
        origin=origin,
        overwrite=True,
    )
    print(f"Saved MRC: {output_mrc}, shape={grid.shape}")
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    cfg = _load_config(args.i)
    task = str(_require(cfg, "task", "root"))
    if task == "inspect":
        return _cli_inspect(cfg)
    if task == "to-pc":
        return _cli_to_pc(cfg)
    if task == "to-mrc":
        return _cli_to_mrc(cfg)
    raise ValueError("Unknown task. Expected one of: inspect, to-pc, to-mrc.")


if __name__ == "__main__":
    raise SystemExit(main())
