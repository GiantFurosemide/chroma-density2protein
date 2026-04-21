"""Orthogonal central-slice QC figures for final MRC density and matching point-cloud slabs."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

Vec3 = Tuple[float, float, float]


def _normalize_voxel_size(voxel_size: Sequence[float] | float) -> Vec3:
    if np.isscalar(voxel_size):
        value = float(voxel_size)
        return (value, value, value)
    out = np.asarray(voxel_size, dtype=np.float64).reshape(-1)
    if out.size == 1:
        v = float(out.item())
        return (v, v, v)
    if out.size != 3:
        raise ValueError("voxel_size must be a float or a 3-value sequence (x, y, z).")
    if np.any(out <= 0):
        raise ValueError("voxel_size values must be positive.")
    return (float(out[0]), float(out[1]), float(out[2]))


def _normalize_origin(origin: Optional[Sequence[float] | None]) -> Vec3:
    if origin is None:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(origin, dtype=np.float64).reshape(-1)
    if arr.size != 3:
        raise ValueError("origin must be a 3-value sequence (x, y, z).")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def plot_final_density_orthogonal(
    arr_zyx: np.ndarray,
    voxel_size: Sequence[float] | float,
    origin: Optional[Sequence[float] | None],
    out_path: str | Path,
    dpi: int = 120,
) -> None:
    """Save a 1×3 figure: XY, XZ, YZ slices through the integer center of the (Z,Y,X) grid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(arr_zyx, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("arr_zyx must have shape (Z, Y, X).")
    nz, ny, nx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    vx, vy, vz = _normalize_voxel_size(voxel_size)
    ox, oy, oz = _normalize_origin(origin)

    iz, iy, ix = nz // 2, ny // 2, nx // 2
    s_xy = arr[iz, :, :]  # (ny, nx)
    s_xz = arr[:, iy, :]  # (nz, nx)
    s_yz = arr[:, :, ix]  # (nz, ny)

    combined = np.concatenate([s_xy.ravel(), s_xz.ravel(), s_yz.ravel()])
    vmin = float(np.percentile(combined, 1.0))
    vmax = float(np.percentile(combined, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # XY @ fixed z: rows=Y, cols=X
    extent_xy = (ox, ox + nx * vx, oy, oy + ny * vy)
    im0 = axes[0].imshow(
        s_xy,
        origin="lower",
        extent=extent_xy,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[0].set_xlabel("x (Å)")
    axes[0].set_ylabel("y (Å)")
    axes[0].set_title(f"XY @ z index {iz}/{nz - 1}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # XZ @ fixed y: rows=Z, cols=X
    extent_xz = (ox, ox + nx * vx, oz, oz + nz * vz)
    im1 = axes[1].imshow(
        s_xz,
        origin="lower",
        extent=extent_xz,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_xlabel("x (Å)")
    axes[1].set_ylabel("z (Å)")
    axes[1].set_title(f"XZ @ y index {iy}/{ny - 1}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # YZ @ fixed x: rows=Z, cols=Y
    extent_yz = (oy, oy + ny * vy, oz, oz + nz * vz)
    im2 = axes[2].imshow(
        s_yz,
        origin="lower",
        extent=extent_yz,
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    axes[2].set_xlabel("y (Å)")
    axes[2].set_ylabel("z (Å)")
    axes[2].set_title(f"YZ @ x index {ix}/{nx - 1}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Final processed mask volume — orthogonal central slices")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_points_orthogonal_slabs(
    points: np.ndarray,
    grid_shape_zyx: Sequence[int],
    voxel_size: Sequence[float] | float,
    origin: Optional[Sequence[float] | None],
    out_path: str | Path,
    dpi: int = 120,
) -> None:
    """Three scatter panels: points in thin slabs around the grid box mid-planes (matches density extent)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3).")
    if pts.shape[0] == 0:
        raise ValueError("points must not be empty.")

    nz, ny, nx = int(grid_shape_zyx[0]), int(grid_shape_zyx[1]), int(grid_shape_zyx[2])
    vx, vy, vz = _normalize_voxel_size(voxel_size)
    ox, oy, oz = _normalize_origin(origin)

    # Physical box aligned with corner-based voxel indices (same as mrc_to_point_cloud).
    x_max = ox + nx * vx
    y_max = oy + ny * vy
    z_max = oz + nz * vz
    xc = ox + 0.5 * nx * vx
    yc = oy + 0.5 * ny * vy
    zc = oz + 0.5 * nz * vz

    def _slab_mask_xy() -> np.ndarray:
        return np.abs(pts[:, 2] - zc) <= 0.5 * max(vz, 1e-6)

    def _slab_mask_xz() -> np.ndarray:
        return np.abs(pts[:, 1] - yc) <= 0.5 * max(vy, 1e-6)

    def _slab_mask_yz() -> np.ndarray:
        return np.abs(pts[:, 0] - xc) <= 0.5 * max(vx, 1e-6)

    m_xy, m_xz, m_yz = _slab_mask_xy(), _slab_mask_xz(), _slab_mask_yz()

    # If a slab is empty, widen once (1.5× voxel half-thickness).
    if not np.any(m_xy):
        m_xy = np.abs(pts[:, 2] - zc) <= 0.75 * max(vz, 1e-6)
    if not np.any(m_xz):
        m_xz = np.abs(pts[:, 1] - yc) <= 0.75 * max(vy, 1e-6)
    if not np.any(m_yz):
        m_yz = np.abs(pts[:, 0] - xc) <= 0.75 * max(vx, 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].scatter(
        pts[m_xy, 0],
        pts[m_xy, 1],
        s=2,
        alpha=0.5,
        c="C0",
        linewidths=0,
    )
    axes[0].set_xlim(ox, x_max)
    axes[0].set_ylim(oy, y_max)
    axes[0].set_xlabel("x (Å)")
    axes[0].set_ylabel("y (Å)")
    axes[0].set_title(f"XY slab @ z≈{zc:.2f} ({int(np.sum(m_xy))} pts)")
    axes[0].set_aspect("equal", adjustable="box")

    axes[1].scatter(
        pts[m_xz, 0],
        pts[m_xz, 2],
        s=2,
        alpha=0.5,
        c="C0",
        linewidths=0,
    )
    axes[1].set_xlim(ox, x_max)
    axes[1].set_ylim(oz, z_max)
    axes[1].set_xlabel("x (Å)")
    axes[1].set_ylabel("z (Å)")
    axes[1].set_title(f"XZ slab @ y≈{yc:.2f} ({int(np.sum(m_xz))} pts)")
    axes[1].set_aspect("equal", adjustable="box")

    axes[2].scatter(
        pts[m_yz, 1],
        pts[m_yz, 2],
        s=2,
        alpha=0.5,
        c="C0",
        linewidths=0,
    )
    axes[2].set_xlim(oy, y_max)
    axes[2].set_ylim(oz, z_max)
    axes[2].set_xlabel("y (Å)")
    axes[2].set_ylabel("z (Å)")
    axes[2].set_title(f"YZ slab @ x≈{xc:.2f} ({int(np.sum(m_yz))} pts)")
    axes[2].set_aspect("equal", adjustable="box")

    fig.suptitle("Point cloud — orthogonal slabs at grid mid-planes")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_input_density_orthogonal(
    arr_zyx: np.ndarray,
    voxel_size: Sequence[float] | float,
    origin: Optional[Sequence[float] | None],
    out_path: str | Path,
    dpi: int = 120,
) -> None:
    """Central XY/XZ/YZ slices of the original input MRC (continuous density)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(arr_zyx, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("arr_zyx must have shape (Z, Y, X).")
    nz, ny, nx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    vx, vy, vz = _normalize_voxel_size(voxel_size)
    ox, oy, oz = _normalize_origin(origin)

    iz, iy, ix = nz // 2, ny // 2, nx // 2
    s_xy = arr[iz, :, :]
    s_xz = arr[:, iy, :]
    s_yz = arr[:, :, ix]

    combined = np.concatenate([s_xy.ravel(), s_xz.ravel(), s_yz.ravel()])
    vmin = float(np.percentile(combined, 1.0))
    vmax = float(np.percentile(combined, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    extent_xy = (ox, ox + nx * vx, oy, oy + ny * vy)
    im0 = axes[0].imshow(
        s_xy, origin="lower", extent=extent_xy, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[0].set_xlabel("x (Å)")
    axes[0].set_ylabel("y (Å)")
    axes[0].set_title(f"XY @ z index {iz}/{nz - 1}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    extent_xz = (ox, ox + nx * vx, oz, oz + nz * vz)
    im1 = axes[1].imshow(
        s_xz, origin="lower", extent=extent_xz, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[1].set_xlabel("x (Å)")
    axes[1].set_ylabel("z (Å)")
    axes[1].set_title(f"XZ @ y index {iy}/{ny - 1}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    extent_yz = (oy, oy + ny * vy, oz, oz + nz * vz)
    im2 = axes[2].imshow(
        s_yz, origin="lower", extent=extent_yz, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[2].set_xlabel("y (Å)")
    axes[2].set_ylabel("z (Å)")
    axes[2].set_title(f"YZ @ x index {ix}/{nx - 1}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Input MRC density — orthogonal central slices")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_input_density_hard_threshold_orthogonal(
    arr_zyx: np.ndarray,
    threshold: float,
    voxel_size: Sequence[float] | float,
    origin: Optional[Sequence[float] | None],
    out_path: str | Path,
    dpi: int = 120,
) -> None:
    """Input density with values < threshold set to 0 (>= threshold unchanged), shown as central XY/XZ/YZ slices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(arr_zyx, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("arr_zyx must have shape (Z, Y, X).")
    thr = float(threshold)
    masked = np.where(arr >= thr, arr, 0.0).astype(np.float32, copy=False)

    nz, ny, nx = int(masked.shape[0]), int(masked.shape[1]), int(masked.shape[2])
    vx, vy, vz = _normalize_voxel_size(voxel_size)
    ox, oy, oz = _normalize_origin(origin)

    iz, iy, ix = nz // 2, ny // 2, nx // 2
    s_xy = masked[iz, :, :]
    s_xz = masked[:, iy, :]
    s_yz = masked[:, :, ix]

    combined = np.concatenate([s_xy.ravel(), s_xz.ravel(), s_yz.ravel()])
    vmin = float(np.percentile(combined, 1.0))
    vmax = float(np.percentile(combined, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(masked)), float(np.max(masked))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    extent_xy = (ox, ox + nx * vx, oy, oy + ny * vy)
    im0 = axes[0].imshow(
        s_xy, origin="lower", extent=extent_xy, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[0].set_xlabel("x (Å)")
    axes[0].set_ylabel("y (Å)")
    axes[0].set_title(f"XY @ z index {iz}/{nz - 1}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    extent_xz = (ox, ox + nx * vx, oz, oz + nz * vz)
    im1 = axes[1].imshow(
        s_xz, origin="lower", extent=extent_xz, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[1].set_xlabel("x (Å)")
    axes[1].set_ylabel("z (Å)")
    axes[1].set_title(f"XZ @ y index {iy}/{ny - 1}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    extent_yz = (oy, oy + ny * vy, oz, oz + nz * vz)
    im2 = axes[2].imshow(
        s_yz, origin="lower", extent=extent_yz, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
    )
    axes[2].set_xlabel("y (Å)")
    axes[2].set_ylabel("z (Å)")
    axes[2].set_title(f"YZ @ x index {ix}/{nx - 1}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"Input density (values < {thr:g} set to 0) — orthogonal central slices")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_input_mask_orthogonal(
    mask_zyx: np.ndarray,
    voxel_size: Sequence[float] | float,
    origin: Optional[Sequence[float] | None],
    out_path: str | Path,
    dpi: int = 120,
) -> None:
    """Central slices of binary mask (0 = below threshold, 1 = at/above threshold)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    arr = np.asarray(mask_zyx, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("mask_zyx must have shape (Z, Y, X).")
    nz, ny, nx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    vx, vy, vz = _normalize_voxel_size(voxel_size)
    ox, oy, oz = _normalize_origin(origin)

    iz, iy, ix = nz // 2, ny // 2, nx // 2
    s_xy = arr[iz, :, :]
    s_xz = arr[:, iy, :]
    s_yz = arr[:, :, ix]

    cmap = ListedColormap(["#2d2d44", "#f6b26b"])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    extent_xy = (ox, ox + nx * vx, oy, oy + ny * vy)
    im0 = axes[0].imshow(s_xy, origin="lower", extent=extent_xy, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    axes[0].set_xlabel("x (Å)")
    axes[0].set_ylabel("y (Å)")
    axes[0].set_title(f"XY @ z index {iz}/{nz - 1}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, ticks=[0.0, 1.0])

    extent_xz = (ox, ox + nx * vx, oz, oz + nz * vz)
    im1 = axes[1].imshow(s_xz, origin="lower", extent=extent_xz, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].set_xlabel("x (Å)")
    axes[1].set_ylabel("z (Å)")
    axes[1].set_title(f"XZ @ y index {iy}/{ny - 1}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, ticks=[0.0, 1.0])

    extent_yz = (oy, oy + ny * vy, oz, oz + nz * vz)
    im2 = axes[2].imshow(s_yz, origin="lower", extent=extent_yz, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    axes[2].set_xlabel("y (Å)")
    axes[2].set_ylabel("z (Å)")
    axes[2].set_title(f"YZ @ x index {ix}/{nx - 1}")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=[0.0, 1.0])

    fig.suptitle("Binary mask (1 = density ≥ threshold) — orthogonal central slices")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _grid_midplanes_and_slabs(
    grid_shape_zyx: Sequence[int],
    voxel_size: Sequence[float] | float,
    origin: Optional[Sequence[float] | None],
    pts: np.ndarray,
    slab_half: Tuple[float, float, float],
) -> Tuple[
    Tuple[float, float, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    float,
    float,
]:
    nz, ny, nx = int(grid_shape_zyx[0]), int(grid_shape_zyx[1]), int(grid_shape_zyx[2])
    vx, vy, vz = _normalize_voxel_size(voxel_size)
    ox, oy, oz = _normalize_origin(origin)
    x_max = ox + nx * vx
    y_max = oy + ny * vy
    z_max = oz + nz * vz
    xc = ox + 0.5 * nx * vx
    yc = oy + 0.5 * ny * vy
    zc = oz + 0.5 * nz * vz
    hx, hy, hz = slab_half
    m_xy = np.abs(pts[:, 2] - zc) <= hz
    m_xz = np.abs(pts[:, 1] - yc) <= hy
    m_yz = np.abs(pts[:, 0] - xc) <= hx
    if not np.any(m_xy):
        m_xy = np.abs(pts[:, 2] - zc) <= 1.5 * hz
    if not np.any(m_xz):
        m_xz = np.abs(pts[:, 1] - yc) <= 1.5 * hy
    if not np.any(m_yz):
        m_yz = np.abs(pts[:, 0] - xc) <= 1.5 * hx
    return (xc, yc, zc), m_xy, m_xz, m_yz, ox, oy, oz, x_max, y_max, z_max


def plot_overlay_points_on_final_volume(
    arr_zyx: np.ndarray,
    voxel_size: Sequence[float] | float,
    origin: Optional[Sequence[float] | None],
    points: np.ndarray,
    out_path: str | Path,
    dpi: int = 120,
) -> None:
    """imshow of final-grid central slices with point scatter on top (same physical extent)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = np.asarray(arr_zyx, dtype=np.float32)
    pts = np.asarray(points, dtype=np.float64)
    if arr.ndim != 3 or pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("Invalid arr_zyx or points shape.")
    nz, ny, nx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    vx, vy, vz = _normalize_voxel_size(voxel_size)
    ox, oy, oz = _normalize_origin(origin)
    iz, iy, ix = nz // 2, ny // 2, nx // 2
    s_xy = arr[iz, :, :]
    s_xz = arr[:, iy, :]
    s_yz = arr[:, :, ix]

    combined = np.concatenate([s_xy.ravel(), s_xz.ravel(), s_yz.ravel()])
    vmin = float(np.percentile(combined, 1.0))
    vmax = float(np.percentile(combined, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(arr)), float(np.max(arr))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    slab_half = (0.5 * max(vx, 1e-6), 0.5 * max(vy, 1e-6), 0.5 * max(vz, 1e-6))
    _, m_xy, m_xz, m_yz, ox, oy, oz, x_max, y_max, z_max = _grid_midplanes_and_slabs(
        arr.shape, voxel_size, origin, pts, slab_half
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    extent_xy = (ox, ox + nx * vx, oy, oy + ny * vy)
    axes[0].imshow(s_xy, origin="lower", extent=extent_xy, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].scatter(
        pts[m_xy, 0], pts[m_xy, 1], s=10, c="red", alpha=0.85, linewidths=0.4, edgecolors="white", zorder=3
    )
    axes[0].set_xlim(ox, x_max)
    axes[0].set_ylim(oy, y_max)
    axes[0].set_xlabel("x (Å)")
    axes[0].set_ylabel("y (Å)")
    axes[0].set_title("XY: final density + points")

    extent_xz = (ox, ox + nx * vx, oz, oz + nz * vz)
    axes[1].imshow(s_xz, origin="lower", extent=extent_xz, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].scatter(
        pts[m_xz, 0], pts[m_xz, 2], s=10, c="red", alpha=0.85, linewidths=0.4, edgecolors="white", zorder=3
    )
    axes[1].set_xlim(ox, x_max)
    axes[1].set_ylim(oz, z_max)
    axes[1].set_xlabel("x (Å)")
    axes[1].set_ylabel("z (Å)")
    axes[1].set_title("XZ: final density + points")

    extent_yz = (oy, oy + ny * vy, oz, oz + nz * vz)
    axes[2].imshow(s_yz, origin="lower", extent=extent_yz, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[2].scatter(
        pts[m_yz, 1], pts[m_yz, 2], s=10, c="red", alpha=0.85, linewidths=0.4, edgecolors="white", zorder=3
    )
    axes[2].set_xlim(oy, y_max)
    axes[2].set_ylim(oz, z_max)
    axes[2].set_xlabel("y (Å)")
    axes[2].set_ylabel("z (Å)")
    axes[2].set_title("YZ: final density + points")

    fig.suptitle("Overlay: processed mask volume + point cloud (slabs at grid mid-planes)")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_points_on_input_mask(
    mask_zyx: np.ndarray,
    voxel_size_input: Sequence[float] | float,
    origin: Optional[Sequence[float] | None],
    points: np.ndarray,
    out_path: str | Path,
    voxel_size_final_for_slab: Optional[Sequence[float] | float] = None,
    dpi: int = 120,
) -> None:
    """Binary input mask slices with points overlaid; slab half-width is max(input/2, final/2) per axis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    arr = np.asarray(mask_zyx, dtype=np.float32)
    pts = np.asarray(points, dtype=np.float64)
    if arr.ndim != 3 or pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("Invalid mask_zyx or points shape.")

    v_in = _normalize_voxel_size(voxel_size_input)
    if voxel_size_final_for_slab is None:
        v_fin = v_in
    else:
        v_fin = _normalize_voxel_size(voxel_size_final_for_slab)
    # Slab thick enough to include binned point corners when comparing to full-res mask.
    slab_half = (
        max(0.5 * v_in[0], 0.5 * v_fin[0], 1e-6),
        max(0.5 * v_in[1], 0.5 * v_fin[1], 1e-6),
        max(0.5 * v_in[2], 0.5 * v_fin[2], 1e-6),
    )

    nz, ny, nx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    vx, vy, vz = v_in
    ox, oy, oz = _normalize_origin(origin)
    iz, iy, ix = nz // 2, ny // 2, nx // 2
    s_xy = arr[iz, :, :]
    s_xz = arr[:, iy, :]
    s_yz = arr[:, :, ix]

    cmap = ListedColormap(["#2d2d44", "#f6b26b"])
    _, m_xy, m_xz, m_yz, ox, oy, oz, x_max, y_max, z_max = _grid_midplanes_and_slabs(
        arr.shape, voxel_size_input, origin, pts, slab_half
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    extent_xy = (ox, ox + nx * vx, oy, oy + ny * vy)
    axes[0].imshow(s_xy, origin="lower", extent=extent_xy, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    axes[0].scatter(
        pts[m_xy, 0], pts[m_xy, 1], s=8, c="cyan", alpha=0.9, linewidths=0.35, edgecolors="k", zorder=3
    )
    axes[0].set_xlim(ox, x_max)
    axes[0].set_ylim(oy, y_max)
    axes[0].set_xlabel("x (Å)")
    axes[0].set_ylabel("y (Å)")
    axes[0].set_title("XY: input mask + points")

    extent_xz = (ox, ox + nx * vx, oz, oz + nz * vz)
    axes[1].imshow(s_xz, origin="lower", extent=extent_xz, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    axes[1].scatter(
        pts[m_xz, 0], pts[m_xz, 2], s=8, c="cyan", alpha=0.9, linewidths=0.35, edgecolors="k", zorder=3
    )
    axes[1].set_xlim(ox, x_max)
    axes[1].set_ylim(oz, oz + nz * vz)
    axes[1].set_xlabel("x (Å)")
    axes[1].set_ylabel("z (Å)")
    axes[1].set_title("XZ: input mask + points")

    extent_yz = (oy, oy + ny * vy, oz, oz + nz * vz)
    axes[2].imshow(s_yz, origin="lower", extent=extent_yz, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    axes[2].scatter(
        pts[m_yz, 1], pts[m_yz, 2], s=8, c="cyan", alpha=0.9, linewidths=0.35, edgecolors="k", zorder=3
    )
    axes[2].set_xlim(oy, y_max)
    axes[2].set_ylim(oz, oz + nz * vz)
    axes[2].set_xlabel("y (Å)")
    axes[2].set_ylabel("z (Å)")
    axes[2].set_title("YZ: input mask + points")

    fig.suptitle(
        "Overlay: full-res binary mask + points (slab half-width max(v_in/2, v_final/2) per axis)"
    )
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
