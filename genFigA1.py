#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Appendix empirical evidence for the "density prior":
Compute local density N_eps(p_i) and evaluate, per dataset, whether field points
tend to have higher density than road points.

This script produces TWO figures (trajectory-level aggregation):
  1) density_traj_mean_bar.pdf   : per-dataset mean of per-trajectory mean densities (road vs field)
  2) density_traj_median_bar.pdf : per-dataset mean of per-trajectory median densities (road vs field)

It also logs diagnostics:
  - point-level mean/median/p95/p99 for road vs field (tail heaviness)
  - per-trajectory summary counts and winsorized means
  - fraction of trajectories where field > road (for mean/median)
  - time-interval (dt) & sampling-frequency (f) diagnostics
  - "counterexample" trajectories where field density is NOT higher than road density
    (for mean and/or median), to address reviewer-raised exceptions.

NOTE:
- harvest time parsing is skipped; dt fixed to 2s (f=0.5Hz) as requested.

Datasets (folders under --root):
  corn/, wheat/, paddy/, tractor/, harvest/
Each Excel file = one trajectory.

IMPORTANT:
- No column auto-detection. You must provide correct per-dataset column names.
- Label convention: value == --field_value is FIELD (y=1), else ROAD (y=0).

EPS UPDATED (field-oriented eps_field_opt from your latest expanded sweep run):
  wheat   : 221.150808 m
  corn    : 132.893344 m
  paddy   : 362.079318 m
  tractor : 159.638744 m
  harvest : 70.515496  m

Outputs are saved under: (script_dir / --outdir)
"""

import os
import glob
import argparse
import logging
import datetime
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from tqdm import tqdm

EARTH_RADIUS_M = 6371008.8  # meters


# -----------------------------
# Core computation
# -----------------------------

def compute_density(lat_deg: np.ndarray, lon_deg: np.ndarray, eps_m: float) -> np.ndarray:
    """Compute N_eps(p_i) within epsilon meters using haversine BallTree (exclude itself)."""
    pts = np.vstack([np.deg2rad(lat_deg), np.deg2rad(lon_deg)]).T
    tree = BallTree(pts, metric="haversine")
    r = float(eps_m) / EARTH_RADIUS_M
    ind = tree.query_radius(pts, r=r)
    return np.array([len(i) - 1 for i in ind], dtype=int)


def summarize_tail(x: np.ndarray) -> Dict[str, float]:
    """Return robust summary stats for tail-effect diagnosis."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {
            "n": 0, "mean": float("nan"), "median": float("nan"),
            "p95": float("nan"), "p99": float("nan"),
            "p99_over_median": float("nan")
        }
    mean = float(np.mean(x))
    med = float(np.median(x))
    p95 = float(np.quantile(x, 0.95))
    p99 = float(np.quantile(x, 0.99))
    p99_over_median = float(p99 / (med + 1e-9))
    return {
        "n": int(x.size),
        "mean": mean,
        "median": med,
        "p95": p95,
        "p99": p99,
        "p99_over_median": p99_over_median
    }


def winsorized_mean(x: np.ndarray, upper_q: float = 0.99) -> float:
    """Winsorize upper tail at given quantile, then compute mean."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    cap = float(np.quantile(x, upper_q))
    x2 = np.minimum(x, cap)
    return float(np.mean(x2))


# -----------------------------
# Time parsing utilities
# -----------------------------

def parse_time_to_seconds(t: pd.Series) -> Optional[np.ndarray]:
    """
    Convert a time column to seconds (float, relative to its minimum).
    Supports:
      - numeric seconds
      - datetime-like strings
      - time-of-day strings like "07:18:32" (handled via to_timedelta fallback)
    Returns None if parsing fails.
    """
    if t is None:
        return None

    # 1) numeric fast path
    if pd.api.types.is_numeric_dtype(t):
        arr = pd.to_numeric(t, errors="coerce").to_numpy(dtype=float)
        if np.isfinite(arr).sum() >= max(10, int(0.1 * len(arr))):
            arr = arr - np.nanmin(arr)
            return arr

    # 2) try datetime parse
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tt = pd.to_datetime(t, errors="coerce")

    if tt.notna().sum() >= max(10, int(0.1 * len(t))):
        s = (tt - tt.min()).dt.total_seconds().to_numpy(dtype=float)
        return s

    # 3) try timedelta parse for "HH:MM:SS" / "MM:SS" etc.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        td = pd.to_timedelta(t, errors="coerce")

    if td.notna().sum() >= max(10, int(0.1 * len(t))):
        s = (td - td.min()).dt.total_seconds().to_numpy(dtype=float)
        return s

    return None


def compute_dt_from_tsec(tsec: np.ndarray) -> np.ndarray:
    """Given per-point time in seconds, compute positive reasonable deltas (sorted)."""
    tsec = np.asarray(tsec, dtype=float)
    tsec = tsec[np.isfinite(tsec)]
    if tsec.size < 3:
        return np.array([], dtype=float)
    tt = np.sort(tsec)
    dt = np.diff(tt)
    dt = dt[np.isfinite(dt)]
    dt = dt[(dt > 0.0) & (dt <= 3600.0)]
    return dt


def dt_to_f_hz(dt_median: float) -> float:
    if not np.isfinite(dt_median) or dt_median <= 0:
        return float("nan")
    return float(1.0 / dt_median)


# -----------------------------
# Data loading per dataset
# -----------------------------

def load_dataset_with_traj_stats(
    dataset_name: str,
    folder: str,
    eps_m: float,
    time_col: Optional[str],
    lat_col: str,
    lon_col: str,
    label_col: str,
    field_value: int,
    max_files: int | None,
    logger: logging.Logger,
    min_points: int = 5,
    min_class_points_per_traj: int = 5,
    harvest_fixed_dt_sec: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    Returns:
      dens_all: concatenated per-point densities across all valid trajectories
      y_all   : concatenated per-point labels (field=1, road=0)
      traj_rows: list of per-trajectory summary dicts
    """
    files = sorted(glob.glob(os.path.join(folder, "*.xls*")))
    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise RuntimeError(f"No Excel files found in folder: {folder}")

    logger.info(f"[{folder}] total files: {len(files)}")

    dens_all, y_all = [], []
    traj_rows: List[Dict] = []

    dt_road_all: List[np.ndarray] = []
    dt_field_all: List[np.ndarray] = []

    for fp in tqdm(files, desc=f"{folder}", unit="file"):
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            logger.warning(f"[{folder}] failed to read {fp}: {e}")
            continue

        needed = [lat_col, lon_col, label_col]
        if time_col is not None:
            needed.append(time_col)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            logger.warning(
                f"[{folder}] {os.path.basename(fp)} missing columns {missing}. "
                f"Available={df.columns.tolist()}"
            )
            continue

        cols = [lat_col, lon_col, label_col] + ([time_col] if time_col is not None else [])
        d = df[cols].copy().dropna()
        if len(d) < min_points:
            logger.info(f"[{folder}] {os.path.basename(fp)} skipped (<{min_points} points)")
            continue

        lat = pd.to_numeric(d[lat_col], errors="coerce").to_numpy()
        lon = pd.to_numeric(d[lon_col], errors="coerce").to_numpy()
        lab = pd.to_numeric(d[label_col], errors="coerce").to_numpy()

        m = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(lab)
        lat, lon, lab = lat[m], lon[m], lab[m]
        if lat.size < min_points:
            logger.info(f"[{folder}] {os.path.basename(fp)} skipped after cleaning (<{min_points})")
            continue

        dens = compute_density(lat, lon, eps_m)
        y = (lab.astype(int) == int(field_value)).astype(int)

        d_field = dens[y == 1]
        d_road = dens[y == 0]

        dt_field_med = float("nan")
        dt_road_med = float("nan")

        if time_col is not None:
            if dataset_name.lower() == "harvest":
                if d_field.size >= 2:
                    dt_field = np.full(int(d_field.size - 1), harvest_fixed_dt_sec, dtype=float)
                    dt_field_all.append(dt_field)
                    dt_field_med = float(harvest_fixed_dt_sec)
                if d_road.size >= 2:
                    dt_road = np.full(int(d_road.size - 1), harvest_fixed_dt_sec, dtype=float)
                    dt_road_all.append(dt_road)
                    dt_road_med = float(harvest_fixed_dt_sec)
            else:
                tsec = parse_time_to_seconds(d[time_col])
                if tsec is not None:
                    tsec2 = np.asarray(tsec)[m]
                    t_field = tsec2[y == 1]
                    t_road = tsec2[y == 0]

                    dt_field = compute_dt_from_tsec(t_field)
                    dt_road = compute_dt_from_tsec(t_road)

                    if dt_field.size > 0:
                        dt_field_all.append(dt_field)
                        dt_field_med = float(np.median(dt_field))
                    if dt_road.size > 0:
                        dt_road_all.append(dt_road)
                        dt_road_med = float(np.median(dt_road))

        dens_all.append(dens)
        y_all.append(y)

        mean_field = float(np.mean(d_field)) if d_field.size > 0 else float("nan")
        mean_road = float(np.mean(d_road)) if d_road.size > 0 else float("nan")
        median_field = float(np.median(d_field)) if d_field.size > 0 else float("nan")
        median_road = float(np.median(d_road)) if d_road.size > 0 else float("nan")

        row = {
            "dataset": folder,
            "file": os.path.basename(fp),
            "n_total": int(dens.size),
            "n_field": int(d_field.size),
            "n_road": int(d_road.size),

            "mean_field": mean_field,
            "mean_road": mean_road,
            "median_field": median_field,
            "median_road": median_road,

            "delta_mean_field_minus_road": float(mean_field - mean_road) if (np.isfinite(mean_field) and np.isfinite(mean_road)) else float("nan"),
            "delta_median_field_minus_road": float(median_field - median_road) if (np.isfinite(median_field) and np.isfinite(median_road)) else float("nan"),

            "wmean_field_99": winsorized_mean(d_field, 0.99) if d_field.size > 0 else float("nan"),
            "wmean_road_99": winsorized_mean(d_road, 0.99) if d_road.size > 0 else float("nan"),

            "dt_median_field_s": dt_field_med,
            "dt_median_road_s": dt_road_med,
            "f_field_hz": dt_to_f_hz(dt_field_med),
            "f_road_hz": dt_to_f_hz(dt_road_med),
        }

        row["usable_for_traj_level"] = bool(
            (d_field.size >= min_class_points_per_traj) and (d_road.size >= min_class_points_per_traj)
        )
        row["counterexample_mean"] = bool(np.isfinite(mean_field) and np.isfinite(mean_road) and (mean_field <= mean_road))
        row["counterexample_median"] = bool(np.isfinite(median_field) and np.isfinite(median_road) and (median_field <= median_road))

        traj_rows.append(row)

    if len(dens_all) == 0:
        raise RuntimeError(f"All trajectories filtered out in {folder}")

    dens_all_cat = np.concatenate(dens_all)
    y_all_cat = np.concatenate(y_all)

    logger.info(f"[{folder}] valid trajectories processed: {len(traj_rows)}")
    logger.info(f"[{folder}] total points: {len(y_all_cat)}")

    def _summ_dt(dt_list: List[np.ndarray]) -> Tuple[float, float, int]:
        if not dt_list:
            return float("nan"), float("nan"), 0
        dt = np.concatenate(dt_list)
        dt = dt[np.isfinite(dt)]
        if dt.size == 0:
            return float("nan"), float("nan"), 0
        dt_med = float(np.median(dt))
        return dt_med, dt_to_f_hz(dt_med), int(dt.size)

    dt_med_field, f_field, n_dt_field = _summ_dt(dt_field_all)
    dt_med_road, f_road, n_dt_road = _summ_dt(dt_road_all)

    logger.info(f"[{folder}][TIME] field: dt_median={dt_med_field:.3f}s, f={f_field:.4f}Hz, n_dt={n_dt_field}")
    logger.info(f"[{folder}][TIME] road : dt_median={dt_med_road:.3f}s, f={f_road:.4f}Hz, n_dt={n_dt_road}")

    return dens_all_cat, y_all_cat, traj_rows


# -----------------------------
# Plotting helpers
# -----------------------------

def plot_grouped_bar(
    ds_list: List[str],
    road_vals: List[float],
    field_vals: List[float],
    road_err: List[float] | None,
    field_err: List[float] | None,
    ylabel: str,
    title: str,
    out_path: str,
    annotate: bool = True
):
    x = np.arange(len(ds_list))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.bar(x - width/2, road_vals, width, yerr=road_err, capsize=4 if road_err is not None else 0, label="road")
    ax.bar(x + width/2, field_vals, width, yerr=field_err, capsize=4 if field_err is not None else 0, label="field")

    ax.set_xticks(x)
    ax.set_xticklabels(ds_list)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.suptitle(title, fontsize=13)

    if annotate:
        for i in range(len(ds_list)):
            ax.text(x[i] - width/2, road_vals[i], f"{road_vals[i]:.1f}", ha="center", va="bottom", fontsize=9)
            ax.text(x[i] + width/2, field_vals[i], f"{field_vals[i]:.1f}", ha="center", va="bottom", fontsize=9)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="/home/ubuntu/forDataSet/AgriTraj",
                        help="Dataset root containing subfolders corn/wheat/paddy/tractor/harvest")
    parser.add_argument("--outdir", type=str, default="pic",
                        help="Output directory name under script directory (created if not exists)")

    # eps params (meters) - UPDATED to eps_field_opt_m from your latest expanded sweep
    parser.add_argument("--wheat_eps", type=float, default=221.150808)
    parser.add_argument("--corn_eps", type=float, default=132.893344)
    parser.add_argument("--paddy_eps", type=float, default=362.079318)
    parser.add_argument("--tractor_eps", type=float, default=159.638744)
    parser.add_argument("--harvest_eps", type=float, default=70.515496)

    parser.add_argument("--field_value", type=int, default=1,
                        help="Label value indicating FIELD class; others treated as road")

    parser.add_argument("--max_files_per_dataset", type=int, default=None)

    parser.add_argument("--min_class_points_per_traj", type=int, default=5,
                        help="A trajectory is used for trajectory-level stats only if both classes have at least this many points.")

    parser.add_argument("--harvest_fixed_dt", type=float, default=2.0,
                        help="For harvest dataset only: force sampling interval dt (seconds), ignore timestamp parsing.")

    # Explicit column names per dataset (time + lat + lon + label)
    parser.add_argument("--corn_time_col", type=str, default="time")
    parser.add_argument("--corn_lat_col", type=str, default="latitude")
    parser.add_argument("--corn_lon_col", type=str, default="longitude")
    parser.add_argument("--corn_label_col", type=str, default="tags")

    parser.add_argument("--wheat_time_col", type=str, default="时间")
    parser.add_argument("--wheat_lat_col", type=str, default="纬度")
    parser.add_argument("--wheat_lon_col", type=str, default="经度")
    parser.add_argument("--wheat_label_col", type=str, default="标签")

    parser.add_argument("--paddy_time_col", type=str, default="时间")
    parser.add_argument("--paddy_lat_col", type=str, default="纬度")
    parser.add_argument("--paddy_lon_col", type=str, default="经度")
    parser.add_argument("--paddy_label_col", type=str, default="标记")

    parser.add_argument("--tractor_time_col", type=str, default="timestamp")
    parser.add_argument("--tractor_lat_col", type=str, default="latitude")
    parser.add_argument("--tractor_lon_col", type=str, default="longitude")
    parser.add_argument("--tractor_label_col", type=str, default="type")

    parser.add_argument("--harvest_time_col", type=str, default="timestamp")
    parser.add_argument("--harvest_lat_col", type=str, default="lat")
    parser.add_argument("--harvest_lon_col", type=str, default="long")
    parser.add_argument("--harvest_label_col", type=str, default="type")

    args = parser.parse_args()

    warnings.filterwarnings("ignore", message="Could not infer format.*")
    warnings.filterwarnings("ignore", message="The argument 'infer_datetime_format' is deprecated.*")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(SCRIPT_DIR, args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    out_mean = os.path.join(out_dir, "density_traj_mean_bar.pdf")
    out_median = os.path.join(out_dir, "density_traj_median_bar.pdf")
    out_csv = os.path.join(out_dir, "density_traj_level_stats.csv")
    out_counter_csv = os.path.join(out_dir, "density_traj_counterexamples.csv")
    log_path = os.path.join(out_dir, "density_A1.log")

    logging.basicConfig(
        filename=log_path,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("DensityA1")

    logger.info("\n" + "=" * 90)
    logger.info("New run started at %s", datetime.datetime.now().isoformat())
    logger.info("Arguments: %s", vars(args))
    logger.info("Script dir: %s", SCRIPT_DIR)
    logger.info("Output dir: %s", out_dir)
    logger.info("=" * 90)
    logger.info("=== Trajectory-level Density Evidence Generation Started ===")

    os.chdir(args.root)

    ds_list = ["corn", "wheat", "paddy", "tractor", "harvest"]
    eps = {
        "corn": args.corn_eps,
        "wheat": args.wheat_eps,
        "paddy": args.paddy_eps,
        "tractor": args.tractor_eps,
        "harvest": args.harvest_eps,
    }
    colcfg = {
        "corn": (args.corn_time_col, args.corn_lat_col, args.corn_lon_col, args.corn_label_col),
        "wheat": (args.wheat_time_col, args.wheat_lat_col, args.wheat_lon_col, args.wheat_label_col),
        "paddy": (args.paddy_time_col, args.paddy_lat_col, args.paddy_lon_col, args.paddy_label_col),
        "tractor": (args.tractor_time_col, args.tractor_lat_col, args.tractor_lon_col, args.tractor_label_col),
        "harvest": (args.harvest_time_col, args.harvest_lat_col, args.harvest_lon_col, args.harvest_label_col),
    }

    all_traj_rows: List[Dict] = []

    for name in ds_list:
        time_col, lat_col, lon_col, lab_col = colcfg[name]
        logger.info(f"Processing dataset: {name} | eps={eps[name]} m | cols=({time_col},{lat_col},{lon_col},{lab_col})")

        dens_all, y_all, traj_rows = load_dataset_with_traj_stats(
            dataset_name=name,
            folder=name,
            eps_m=eps[name],
            time_col=time_col,
            lat_col=lat_col,
            lon_col=lon_col,
            label_col=lab_col,
            field_value=args.field_value,
            max_files=args.max_files_per_dataset,
            logger=logger,
            min_points=5,
            min_class_points_per_traj=args.min_class_points_per_traj,
            harvest_fixed_dt_sec=args.harvest_fixed_dt
        )
        all_traj_rows.extend(traj_rows)

        d_field = dens_all[y_all == 1]
        d_road = dens_all[y_all == 0]

        s_field = summarize_tail(d_field)
        s_road = summarize_tail(d_road)

        logger.info(
            f"[{name}][POINT] field: n={s_field['n']}, mean={s_field['mean']:.3f}, median={s_field['median']:.3f}, "
            f"p95={s_field['p95']:.3f}, p99={s_field['p99']:.3f}, p99/med={s_field['p99_over_median']:.3f}"
        )
        logger.info(
            f"[{name}][POINT] road : n={s_road['n']}, mean={s_road['mean']:.3f}, median={s_road['median']:.3f}, "
            f"p95={s_road['p95']:.3f}, p99={s_road['p99']:.3f}, p99/med={s_road['p99_over_median']:.3f}"
        )

        tail_flag = (s_road["mean"] > s_field["mean"]) and (s_road["median"] <= s_field["median"])
        if tail_flag:
            logger.info(f"[{name}] Tail-effect suspected: road mean > field mean but road median <= field median.")
        else:
            logger.info(f"[{name}] Tail-effect flag: {tail_flag}")

    traj_df = pd.DataFrame(all_traj_rows)
    traj_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Trajectory-level stats CSV saved to {out_csv}")

    usable = traj_df["usable_for_traj_level"] == True
    traj_use = traj_df.loc[usable].copy()
    logger.info(f"Total trajectories (rows): {len(traj_df)}; usable for traj-level: {len(traj_use)} "
                f"(min_class_points_per_traj={args.min_class_points_per_traj})")

    # --- Counterexamples CSV (address reviewer exceptions) ---
    ce = traj_use[(traj_use["counterexample_mean"] == True) | (traj_use["counterexample_median"] == True)].copy()
    ce = ce.sort_values(["dataset", "counterexample_mean", "counterexample_median"], ascending=[True, False, False])
    ce.to_csv(out_counter_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Counterexample trajectories CSV saved to {out_counter_csv} (rows={len(ce)})")

    # --- Aggregate + plot ---
    mean_road_vals, mean_field_vals = [], []
    median_road_vals, median_field_vals = [], []
    mean_road_se, mean_field_se = [], []
    median_road_se, median_field_se = [], []
    frac_mean_field_gt_road, frac_median_field_gt_road = [], []

    for name in ds_list:
        sub = traj_use[traj_use["dataset"] == name]
        if len(sub) == 0:
            raise RuntimeError(
                f"No usable trajectories for dataset {name}. "
                f"Consider lowering --min_class_points_per_traj (current={args.min_class_points_per_traj})."
            )

        r_mean = sub["mean_road"].to_numpy(dtype=float)
        f_mean = sub["mean_field"].to_numpy(dtype=float)
        r_med = sub["median_road"].to_numpy(dtype=float)
        f_med = sub["median_field"].to_numpy(dtype=float)

        mean_road_vals.append(float(np.mean(r_mean)))
        mean_field_vals.append(float(np.mean(f_mean)))
        median_road_vals.append(float(np.mean(r_med)))
        median_field_vals.append(float(np.mean(f_med)))

        mean_road_se.append(float(np.std(r_mean, ddof=1) / np.sqrt(len(r_mean))) if len(r_mean) > 1 else 0.0)
        mean_field_se.append(float(np.std(f_mean, ddof=1) / np.sqrt(len(f_mean))) if len(f_mean) > 1 else 0.0)
        median_road_se.append(float(np.std(r_med, ddof=1) / np.sqrt(len(r_med))) if len(r_med) > 1 else 0.0)
        median_field_se.append(float(np.std(f_med, ddof=1) / np.sqrt(len(f_med))) if len(f_med) > 1 else 0.0)

        frac_mean_field_gt_road.append(float(np.mean(f_mean > r_mean)))
        frac_median_field_gt_road.append(float(np.mean(f_med > r_med)))

        # quick counterexample counts for log (useful for rebuttal text)
        n_ce_mean = int(np.sum(sub["counterexample_mean"].to_numpy(dtype=bool)))
        n_ce_med = int(np.sum(sub["counterexample_median"].to_numpy(dtype=bool)))

        logger.info(
            f"[{name}][TRAJ] n_traj={len(sub)} | "
            f"MEAN: field={mean_field_vals[-1]:.3f} (SE={mean_field_se[-1]:.3f}), "
            f"road={mean_road_vals[-1]:.3f} (SE={mean_road_se[-1]:.3f}), "
            f"frac(field>road)={frac_mean_field_gt_road[-1]:.3f}, "
            f"counterexamples={n_ce_mean} | "
            f"MEDIAN: field={median_field_vals[-1]:.3f} (SE={median_field_se[-1]:.3f}), "
            f"road={median_road_vals[-1]:.3f} (SE={median_road_se[-1]:.3f}), "
            f"frac(field>road)={frac_median_field_gt_road[-1]:.3f}, "
            f"counterexamples={n_ce_med}"
        )

        wm_r = sub["wmean_road_99"].to_numpy(dtype=float)
        wm_f = sub["wmean_field_99"].to_numpy(dtype=float)
        logger.info(
            f"[{name}][TRAJ] winsorized-mean@99%: field={float(np.mean(wm_f)):.3f}, road={float(np.mean(wm_r)):.3f}"
        )

        # time summary at traj-level (median over trajectories)
        dt_rf = sub["dt_median_road_s"].to_numpy(dtype=float)
        dt_ff = sub["dt_median_field_s"].to_numpy(dtype=float)
        dt_rf = dt_rf[np.isfinite(dt_rf)]
        dt_ff = dt_ff[np.isfinite(dt_ff)]
        if dt_rf.size > 0:
            logger.info(f"[{name}][TIME-TRAJ] road : median(dt_median_traj)={float(np.median(dt_rf)):.3f}s, "
                        f"median(f_traj)={float(np.median(1.0/np.maximum(dt_rf, 1e-9))):.4f}Hz, n_traj_dt={dt_rf.size}")
        if dt_ff.size > 0:
            logger.info(f"[{name}][TIME-TRAJ] field: median(dt_median_traj)={float(np.median(dt_ff)):.3f}s, "
                        f"median(f_traj)={float(np.median(1.0/np.maximum(dt_ff, 1e-9))):.4f}Hz, n_traj_dt={dt_ff.size}")

    plot_grouped_bar(
        ds_list=ds_list,
        road_vals=mean_road_vals,
        field_vals=mean_field_vals,
        road_err=mean_road_se,
        field_err=mean_field_se,
        ylabel=r"Trajectory-level mean density  $\mathbb{E}_t[\; \mathbb{E}(N_{\varepsilon})\;]$",
        title="Dataset-level comparison (trajectory-level mean): road vs field",
        out_path=out_mean,
        annotate=True
    )

    plot_grouped_bar(
        ds_list=ds_list,
        road_vals=median_road_vals,
        field_vals=median_field_vals,
        road_err=median_road_se,
        field_err=median_field_se,
        ylabel=r"Trajectory-level median density  $\mathbb{E}_t[\; \mathrm{median}(N_{\varepsilon})\;]$",
        title="Dataset-level comparison (trajectory-level median): road vs field",
        out_path=out_median,
        annotate=True
    )

    logger.info(f"Figure saved to {out_mean}")
    logger.info(f"Figure saved to {out_median}")
    logger.info("=== Finished ===")

    print(f"Saved trajectory-level MEAN bar chart   -> {out_mean}")
    print(f"Saved trajectory-level MEDIAN bar chart -> {out_median}")
    print(f"Saved trajectory-level stats CSV        -> {out_csv}")
    print(f"Saved counterexamples CSV              -> {out_counter_csv}")
    print(f"Log written to                          -> {log_path}")


if __name__ == "__main__":
    main()
