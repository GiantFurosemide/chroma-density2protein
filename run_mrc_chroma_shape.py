"""MRC → binary mask → point cloud → Chroma shape design (project entry, run from repo root)."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from chroma import Chroma, api, conditioners

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mrc_pointcloud as mpc
import qc_orthogonal_plots as qc_plots

# Foreground on binarized / smoothed mask grid inside adaptive_cap (>= this value counts as "on").
ADAPTIVE_MASK_FOREGROUND_THRESHOLD = 0.5


def _cfg_for_report(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    if out.get("api_key"):
        out["api_key"] = "***redacted***"
    return out


def _build_text_summary(report: Dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("MRC → Chroma shape — run report (key parameters)")
    lines.append("=" * 60)
    lines.append(f"Config file: {report.get('config_source', '')}")
    lines.append("")

    r = report.get("resolved") or {}
    lines.append("[Threshold / input map]")
    lines.append(f"  threshold value: {r.get('threshold')}")
    lines.append(f"  threshold source: {r.get('threshold_source')}")
    if r.get("threshold_source") == "quantile_derived":
        lines.append(f"  threshold_quantile (YAML): {r.get('threshold_quantile')}")
    lines.append(f"  input MRC: {r.get('input_mrc')}")
    lines.append(f"  MRC shape (Z,Y,X): {r.get('mrc_shape_zyx')}")
    lines.append(f"  initial voxel_size (x,y,z): {r.get('initial_voxel_size_xyz')}")
    lines.append(f"  origin (x,y,z): {r.get('origin_xyz')}")
    if r.get("binary_mask_mode"):
        lines.append("  binary mask: density >= threshold → 1, else 0 (threshold above)")
        ap = r.get("adaptive_pooling")
        if ap == "binary_max":
            lines.append(
                "  adaptive downsampling: 2×2×2 max-pool (foreground stays 0/1; only spatial coarsening)"
            )
        else:
            lines.append(
                f"  adaptive foreground on processed grid: >= {r.get('adaptive_selection_threshold')} "
                "(after FFT/mean smoothing)"
            )
    lines.append("")

    ab = report.get("adaptive_binning")
    if ab is not None:
        ap = ab.get("adaptive_pooling", "mean_fft")
        if ap == "binary_max":
            lines.append("[Adaptive binning (2×2×2 max-pool, binary-safe)]")
        else:
            lines.append("[Adaptive binning (FFT low-pass + 2x2x2 mean bin)]")
        lines.append(f"  adaptive_pooling: {ap}")
        lines.append(f"  max_voxels (cap): {ab.get('max_voxels')}")
        lines.append(f"  max_bin_rounds (safety): {ab.get('max_bin_rounds')}")
        lines.append(f"  bin rounds applied: {ab.get('bin_rounds_applied')}")
        lines.append(
            f"  mask voxels == 1 (initial full res, before binning): {ab.get('initial_full_res_mask_ones')}"
        )
        lines.append(
            f"  foreground voxels (final grid, before random subsample): "
            f"{ab.get('final_voxels_ge_threshold_before_subsample')}"
        )
        lines.append(
            f"  under cap without random subsample: {ab.get('selected_count_le_max_voxels_before_subsample')}"
        )
        lines.append(f"  used random subsample to hit cap: {ab.get('used_random_subsample')}")
        lines.append(f"  final shape (Z,Y,X): {ab.get('final_shape_zyx')}")
        lines.append(f"  final voxel_size (x,y,z): {ab.get('final_voxel_size_xyz')}")
        lines.append("")
    else:
        vm = report.get("voxel_selection_non_adaptive") or {}
        lines.append("[Voxel selection (no adaptive binning)]")
        lines.append(f"  mask voxels == 1 (full res): {vm.get('full_res_mask_ones')}")
        lines.append("")

    pc = report.get("point_cloud") or {}
    lines.append("[Point cloud outputs]")
    lines.append(f"  points in saved .npy: {pc.get('n_points_saved')}")
    lines.append(f"  points passed to ShapeConditioner: {pc.get('points_to_conditioner')}")
    lines.append(f"  max_points_conditioner (YAML): {pc.get('max_points_conditioner')}")
    lines.append(f"  conditioner subsample used: {pc.get('conditioner_subsample_used')}")
    lines.append(f"  saved path: {pc.get('saved_npy_path')}")
    lines.append("")

    qv = report.get("qc_visualization")
    if qv:
        lines.append("[QC visualization]")
        lines.append(f"  input density PNG: {qv.get('qc_input_density_png_path')}")
        lines.append(f"  input density (below threshold → 0) PNG: {qv.get('qc_input_density_thr0_png_path')}")
        lines.append(f"  input binary mask PNG: {qv.get('qc_input_mask_png_path')}")
        lines.append(f"  final processed mask MRC: {qv.get('final_density_mrc_path')}")
        lines.append(f"  final density orthogonal PNG: {qv.get('qc_density_png_path')}")
        lines.append(f"  points orthogonal PNG: {qv.get('qc_points_png_path')}")
        lines.append(f"  overlay final + points PNG: {qv.get('qc_overlay_final_png_path')}")
        lines.append(f"  overlay input mask + points PNG: {qv.get('qc_overlay_input_mask_png_path')}")
        lines.append("")

    ch = report.get("chroma")
    lines.append("[Chroma sampling]")
    if ch is None:
        lines.append("  (not finished yet — sampling in progress or skipped)")
    else:
        lines.append(f"  num_residues: {ch.get('num_residues')}")
        lines.append(f"  seed: {ch.get('seed')}")
        lines.append(f"  device: {ch.get('device')}")
        lines.append(f"  output PDB: {ch.get('protein_pdb')}")
        lines.append(f"  output CIF: {ch.get('protein_cif')}")
    lines.append("")
    lines.append(f"Full JSON report: {report.get('artifacts', {}).get('json_report_path', '')}")
    lines.append("=" * 60)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MRC → binary mask → point cloud → Chroma shape design (run from repository root)."
    )
    parser.add_argument(
        "--i",
        default="configs/mrc_chroma_shape.yaml",
        help="Pipeline config YAML (default: configs/mrc_chroma_shape.yaml).",
    )
    args = parser.parse_args()
    config_path = Path(args.i).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{config_path} must contain a YAML object at top level.")

    input_mrc = Path(str(cfg["input_mrc"]))
    if not input_mrc.is_absolute():
        input_mrc = (Path.cwd() / input_mrc).resolve()
    output_dir = Path(str(cfg["output_dir"]))
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_raw = cfg.get("threshold")
    threshold_quantile = float(cfg.get("threshold_quantile", 0.995))
    use_adaptive_bin = bool(cfg.get("use_adaptive_bin", True))
    max_voxels = int(cfg.get("max_voxels", 2000))
    max_bin_rounds = int(cfg.get("max_bin_rounds", 32))
    adaptive_pooling = str(cfg.get("adaptive_pooling", "binary_max")).strip().lower()
    if adaptive_pooling not in ("binary_max", "mean_fft"):
        raise ValueError('adaptive_pooling must be "binary_max" or "mean_fft".')
    subsample_seed = cfg.get("adaptive_subsample_seed", cfg.get("seed", 0))
    num_residues = int(cfg.get("num_residues", 600))
    seed = int(cfg.get("seed", 0))
    api_key = cfg.get("api_key")

    if api_key:
        api.register_key(str(api_key))

    data, voxel_size, origin, header = mpc.read_mrc(str(input_mrc))
    stats = mpc.inspect_mrc(data, quantiles=(0.5, 0.9, 0.95, 0.99, threshold_quantile))

    threshold_source = "explicit_yaml" if threshold_raw is not None else "quantile_derived"
    if threshold_raw is None:
        if use_adaptive_bin:
            warnings.warn(
                "threshold is null: using threshold_quantile for a single threshold value; "
                "prefer setting an explicit threshold for reproducible adaptive binning.",
                stacklevel=1,
            )
        threshold = float(np.quantile(data.astype(np.float64), threshold_quantile))
    else:
        threshold = float(threshold_raw)

    mask_data = (data >= threshold).astype(np.float32)
    initial_full_res_mask_ones = int(np.count_nonzero(mask_data >= ADAPTIVE_MASK_FOREGROUND_THRESHOLD))

    qc_input_density_png = output_dir / "module1_2_15A_input_density_orthogonal.png"
    qc_input_density_thr0_png = (
        output_dir / "module1_2_15A_input_density_ge_threshold_else_zero_orthogonal.png"
    )
    qc_input_mask_png = output_dir / "module1_2_15A_input_mask_orthogonal.png"
    qc_plots.plot_input_density_orthogonal(
        data,
        voxel_size=voxel_size,
        origin=origin,
        out_path=qc_input_density_png,
    )
    qc_plots.plot_input_density_hard_threshold_orthogonal(
        data,
        threshold=threshold,
        voxel_size=voxel_size,
        origin=origin,
        out_path=qc_input_density_thr0_png,
    )
    qc_plots.plot_input_mask_orthogonal(
        mask_data,
        voxel_size=voxel_size,
        origin=origin,
        out_path=qc_input_mask_png,
    )

    adaptive_meta: Optional[Dict[str, Any]] = None
    if use_adaptive_bin:
        points, adaptive_meta, final_density = mpc.mrc_to_point_cloud_adaptive_cap(
            data=mask_data,
            voxel_size=voxel_size,
            origin=origin,
            threshold=ADAPTIVE_MASK_FOREGROUND_THRESHOLD,
            max_voxels=max_voxels,
            max_rounds=max_bin_rounds,
            subsample_seed=None if subsample_seed is None else int(subsample_seed),
            adaptive_pooling=adaptive_pooling,
        )
        final_voxel_size = tuple(float(v) for v in adaptive_meta["final_voxel_size_xyz"])
    else:
        points = mpc.mrc_to_point_cloud(
            data=mask_data,
            voxel_size=voxel_size,
            origin=origin,
            threshold=ADAPTIVE_MASK_FOREGROUND_THRESHOLD,
            max_points=None,
            random_seed=None,
        )
        final_density = mask_data
        final_voxel_size = (float(voxel_size[0]), float(voxel_size[1]), float(voxel_size[2]))

    final_density_mrc = output_dir / "module1_2_15A_final_density.mrc"
    qc_density_png = output_dir / "module1_2_15A_final_density_orthogonal.png"
    qc_points_png = output_dir / "module1_2_15A_points_orthogonal.png"
    qc_overlay_final_png = output_dir / "module1_2_15A_overlay_points_on_final.png"
    qc_overlay_input_mask_png = output_dir / "module1_2_15A_overlay_points_on_input_mask.png"

    mpc.write_mrc(
        str(final_density_mrc),
        final_density,
        voxel_size=final_voxel_size,
        origin=origin,
        overwrite=True,
    )
    qc_plots.plot_final_density_orthogonal(
        final_density,
        voxel_size=final_voxel_size,
        origin=origin,
        out_path=qc_density_png,
    )
    qc_plots.plot_points_orthogonal_slabs(
        points,
        grid_shape_zyx=final_density.shape,
        voxel_size=final_voxel_size,
        origin=origin,
        out_path=qc_points_png,
    )
    qc_plots.plot_overlay_points_on_final_volume(
        final_density,
        voxel_size=final_voxel_size,
        origin=origin,
        points=points,
        out_path=qc_overlay_final_png,
    )
    qc_plots.plot_overlay_points_on_input_mask(
        mask_data,
        voxel_size_input=voxel_size,
        origin=origin,
        points=points,
        out_path=qc_overlay_input_mask_png,
        voxel_size_final_for_slab=final_voxel_size,
    )

    points_npy = output_dir / "module1_2_15A_points.npy"
    np.save(points_npy, points)

    max_points_conditioner = cfg.get("max_points_conditioner")
    conditioner_seed = cfg.get("conditioner_subsample_seed", seed)
    points_for_chroma = points
    conditioner_subsample_used = False
    if max_points_conditioner is not None:
        cap = int(max_points_conditioner)
        if cap <= 0:
            raise ValueError("max_points_conditioner must be positive when set.")
        if points.shape[0] > cap:
            rng = np.random.default_rng(seed=int(conditioner_seed))
            idx = rng.choice(points.shape[0], size=cap, replace=False)
            points_for_chroma = points[idx]
            conditioner_subsample_used = True

    resolved: Dict[str, Any] = {
        "threshold": threshold,
        "threshold_used_for_binarization": threshold,
        "threshold_source": threshold_source,
        "threshold_quantile": threshold_quantile,
        "binary_mask_mode": True,
        "adaptive_selection_threshold": ADAPTIVE_MASK_FOREGROUND_THRESHOLD,
        "input_mrc": str(input_mrc.resolve()),
        "mrc_shape_zyx": [int(data.shape[0]), int(data.shape[1]), int(data.shape[2])],
        "initial_voxel_size_xyz": [float(v) for v in voxel_size],
        "origin_xyz": [float(v) for v in origin],
        "initial_full_res_mask_ones": initial_full_res_mask_ones,
        "adaptive_pooling": adaptive_pooling,
    }

    adaptive_section: Optional[Dict[str, Any]] = None
    voxel_non_adaptive: Optional[Dict[str, Any]] = None
    if use_adaptive_bin and adaptive_meta is not None:
        adaptive_section = {
            "adaptive_pooling": adaptive_meta.get("adaptive_pooling", adaptive_pooling),
            "max_voxels": max_voxels,
            "max_bin_rounds": max_bin_rounds,
            "adaptive_subsample_seed": subsample_seed,
            "threshold_used_for_binarization": threshold,
            "adaptive_selection_threshold": ADAPTIVE_MASK_FOREGROUND_THRESHOLD,
            "initial_full_res_mask_ones": initial_full_res_mask_ones,
            "bin_rounds_applied": adaptive_meta["bin_rounds_applied"],
            "initial_voxels_ge_threshold": adaptive_meta["initial_selected_voxels_ge_threshold"],
            "final_voxels_ge_threshold_before_subsample": adaptive_meta[
                "final_selected_voxels_ge_threshold_before_subsample"
            ],
            "selected_count_le_max_voxels_before_subsample": adaptive_meta[
                "selected_count_le_max_voxels_before_subsample"
            ],
            "used_random_subsample": adaptive_meta["used_random_subsample"],
            "final_shape_zyx": adaptive_meta["final_shape_zyx"],
            "final_voxel_size_xyz": adaptive_meta["final_voxel_size_xyz"],
            "history": adaptive_meta["history"],
            "full_adaptive_meta": adaptive_meta,
        }
    else:
        voxel_non_adaptive = {
            "full_res_mask_ones": initial_full_res_mask_ones,
        }

    point_cloud_section: Dict[str, Any] = {
        "saved_npy_path": str(points_npy.resolve()),
        "n_points_saved": int(points.shape[0]),
        "points_to_conditioner": int(points_for_chroma.shape[0]),
        "max_points_conditioner": max_points_conditioner,
        "conditioner_subsample_seed": conditioner_seed,
        "conditioner_subsample_used": conditioner_subsample_used,
    }

    qc_visualization_section: Dict[str, Any] = {
        "qc_input_density_png_path": str(qc_input_density_png.resolve()),
        "qc_input_density_thr0_png_path": str(qc_input_density_thr0_png.resolve()),
        "qc_input_mask_png_path": str(qc_input_mask_png.resolve()),
        "final_density_mrc_path": str(final_density_mrc.resolve()),
        "qc_density_png_path": str(qc_density_png.resolve()),
        "qc_points_png_path": str(qc_points_png.resolve()),
        "qc_overlay_final_png_path": str(qc_overlay_final_png.resolve()),
        "qc_overlay_input_mask_png_path": str(qc_overlay_input_mask_png.resolve()),
    }

    metrics = {
        "input_mrc": str(input_mrc),
        "shape_zyx": list(data.shape),
        "voxel_size_xyz": list(voxel_size),
        "origin_xyz": list(origin),
        "threshold": threshold,
        "threshold_used_for_binarization": threshold,
        "binary_mask_mode": True,
        "adaptive_selection_threshold": ADAPTIVE_MASK_FOREGROUND_THRESHOLD,
        "threshold_quantile": threshold_quantile,
        "use_adaptive_bin": use_adaptive_bin,
        "adaptive_pooling": adaptive_pooling if use_adaptive_bin else None,
        "max_voxels": max_voxels if use_adaptive_bin else None,
        "max_bin_rounds": max_bin_rounds if use_adaptive_bin else None,
        "adaptive_meta": adaptive_meta,
        "selected_points": int(points.shape[0]),
        "points_for_chroma": int(points_for_chroma.shape[0]),
        "max_points_conditioner": max_points_conditioner,
        "header": header,
        "stats": stats,
    }
    with (output_dir / "module1_2_15A_pointcloud_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")

    json_report_path = output_dir / "mrc_chroma_shape_run_report.json"
    txt_report_path = output_dir / "mrc_chroma_shape_run_report.txt"

    partial_report: Dict[str, Any] = {
        "pipeline": "run_mrc_chroma_shape.py",
        "config_source": str(config_path),
        "config_raw": _cfg_for_report(cfg),
        "resolved": resolved,
        "adaptive_binning": adaptive_section,
        "voxel_selection_non_adaptive": voxel_non_adaptive,
        "point_cloud": point_cloud_section,
        "qc_visualization": qc_visualization_section,
        "header_summary": header,
        "density_stats": stats,
        "artifacts": {
            "json_report_path": str(json_report_path.resolve()),
            "txt_report_path": str(txt_report_path.resolve()),
            "pointcloud_metrics_json": str((output_dir / "module1_2_15A_pointcloud_metrics.json").resolve()),
            "qc_input_density_png_path": str(qc_input_density_png.resolve()),
            "qc_input_density_thr0_png_path": str(qc_input_density_thr0_png.resolve()),
            "qc_input_mask_png_path": str(qc_input_mask_png.resolve()),
            "final_density_mrc_path": str(final_density_mrc.resolve()),
            "qc_density_png_path": str(qc_density_png.resolve()),
            "qc_points_png_path": str(qc_points_png.resolve()),
            "qc_overlay_final_png_path": str(qc_overlay_final_png.resolve()),
            "qc_overlay_input_mask_png_path": str(qc_overlay_input_mask_png.resolve()),
        },
        "chroma": None,
    }

    with txt_report_path.open("w", encoding="utf-8") as f:
        f.write(_build_text_summary(partial_report))

    print(_build_text_summary(partial_report))
    print("[After point cloud] Partial report written; Chroma sampling next...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    chroma = Chroma()
    conditioner = conditioners.ShapeConditioner(
        points_for_chroma,
        chroma.backbone_network.noise_schedule,
        autoscale_num_residues=num_residues,
    ).to(device)

    torch.manual_seed(seed)
    shaped_protein = chroma.sample(chain_lengths=[num_residues], conditioner=conditioner)

    out_pdb = output_dir / "module1_2_15A_shape_design.pdb"
    out_cif = output_dir / "module1_2_15A_shape_design.cif"
    shaped_protein.to(str(out_pdb))
    shaped_protein.to(str(out_cif))

    partial_report["chroma"] = {
        "num_residues": num_residues,
        "seed": seed,
        "device": device,
        "protein_pdb": str(out_pdb.resolve()),
        "protein_cif": str(out_cif.resolve()),
    }

    with json_report_path.open("w", encoding="utf-8") as f:
        json.dump(partial_report, f, indent=2, sort_keys=True)
        f.write("\n")

    with txt_report_path.open("w", encoding="utf-8") as f:
        f.write(_build_text_summary(partial_report))

    print(_build_text_summary(partial_report))
    print(
        json.dumps(
            {
                "json_report": str(json_report_path),
                "txt_report": str(txt_report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
