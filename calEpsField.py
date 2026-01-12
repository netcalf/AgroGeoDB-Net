#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Field-oriented EPS search (wider range):
Find an epsilon (per dataset) such that the density prior holds better:
    field tends to have higher local density than road.

Core idea:
- Use a BASE eps from road knee (eps_base_from_road_knee_m) as a scale anchor.
- Sweep eps = gamma * eps_base over a WIDER gamma range (default 0.5 ~ 8.0, log-spaced).
- For each eps candidate, compute per-trajectory densities and evaluate:
    delta_mean   = mean_t( mean_field - mean_road )
    delta_median = mean_t( median_field - median_road )
    frac_mean_gt   = fraction of trajectories where mean_field > mean_road
    frac_median_gt = fraction of trajectories where median_field > median_road
- Choose eps that maximizes a direction-sensitive score (penalize negative deltas).

Outputs (under script_dir/--outdir):
- eps_field_opt.csv        : best eps per dataset + metrics
- eps_field_sweep.csv      : sweep results per dataset/gamma/eps
- eps_field_sweep_{ds}.pdf : sweep curves (score + deltas + fractions) for each dataset
- eps_field_search.log     : append log

Datasets under --root:
  corn/, wheat/, paddy/, tractor/, harvest/
Each Excel file = one trajectory.

IMPORTANT:
- You must provide correct column names for each dataset.
- Label convention: value == --field_value is FIELD (1), else ROAD (0).

Note:
This script does NOT depend on timestamps (only lat/lon + label).
harvest timestamp issues are irrelevant here.
"""

import os
import glob
import argparse
import logging
import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from tqdm import tqdm

EARTH_RADIUS_M = 6371008.8  # meters


# -----------------------------
# Logging
# -----------------------------

def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    return logging.getLogger("EpsFieldSearch")


# -----------------------------
# Density computation
# -----------------------------

def compute_density(lat_deg: np.ndarray, lon_deg: np.ndarray, eps_m: float) -> np.ndarray:
    """Compute N_eps(p_i) within epsilon meters using haversine BallTree (exclude itself)."""
    pts = np.vstack([np.deg2rad(lat_deg), np.deg2rad(lon_deg)]).T
    tree = BallTree(pts, metric="haversine")
    r = float(eps_m) / EARTH_RADIUS_M
    ind = tree.query_radius(pts, r=r)
    return np.array([len(i) - 1 for i in ind], dtype=int)


def winsorized_mean(x: np.ndarray, upper_q: float = 0.99) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return float("nan")
    cap = float(np.quantile(x, upper_q))
    return float(np.mean(np.minimum(x, cap)))


# -----------------------------
# Trajectory stats for a given eps
# -----------------------------

def load_traj_stats_for_eps(
    folder: str,
    eps_m: float,
    lat_col: str,
    lon_col: str,
    label_col: str,
    field_value: int,
    max_files: Optional[int],
    min_points: int,
    min_class_points_per_traj: int,
    max_points_per_traj: int,
    seed: int,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, int]:
    """
    For a fixed eps_m, compute per-trajectory density stats.

    Returns:
      traj_df: per-trajectory stats (only usable trajectories flagged)
      n_files_total: number of files in folder (after max_files cap)
    """
    files = sorted(glob.glob(os.path.join(folder, "*.xls*")))
    if max_files is not None:
        files = files[:max_files]
    n_files_total = len(files)
    if n_files_total == 0:
        raise RuntimeError(f"No Excel files found in folder: {folder}")

    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    for fp in files:
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            logger.warning(f"[{folder}] failed to read {fp}: {e}")
            continue

        missing = [c for c in (lat_col, lon_col, label_col) if c not in df.columns]
        if missing:
            logger.warning(f"[{folder}] {os.path.basename(fp)} missing columns {missing}")
            continue

        d = df[[lat_col, lon_col, label_col]].copy().dropna()
        if len(d) < min_points:
            continue

        lat = pd.to_numeric(d[lat_col], errors="coerce").to_numpy()
        lon = pd.to_numeric(d[lon_col], errors="coerce").to_numpy()
        lab = pd.to_numeric(d[label_col], errors="coerce").to_numpy()

        ok = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(lab)
        lat, lon, lab = lat[ok], lon[ok], lab[ok]
        if lat.size < min_points:
            continue

        # optional downsample within a trajectory for speed (keeps distribution reasonably)
        if max_points_per_traj > 0 and lat.size > max_points_per_traj:
            idx = rng.choice(lat.size, size=max_points_per_traj, replace=False)
            lat, lon, lab = lat[idx], lon[idx], lab[idx]

        dens = compute_density(lat, lon, eps_m)
        y = (lab.astype(int) == int(field_value)).astype(int)

        d_field = dens[y == 1]
        d_road = dens[y == 0]

        usable = (d_field.size >= min_class_points_per_traj) and (d_road.size >= min_class_points_per_traj)

        rows.append({
            "dataset": folder,
            "file": os.path.basename(fp),
            "n_total": int(dens.size),
            "n_field": int(d_field.size),
            "n_road": int(d_road.size),
            "mean_field": float(np.mean(d_field)) if d_field.size else float("nan"),
            "mean_road": float(np.mean(d_road)) if d_road.size else float("nan"),
            "median_field": float(np.median(d_field)) if d_field.size else float("nan"),
            "median_road": float(np.median(d_road)) if d_road.size else float("nan"),
            "wmean_field_99": winsorized_mean(d_field, 0.99) if d_field.size else float("nan"),
            "wmean_road_99": winsorized_mean(d_road, 0.99) if d_road.size else float("nan"),
            "usable": bool(usable),
        })

    traj_df = pd.DataFrame(rows)
    if len(traj_df) == 0:
        raise RuntimeError(f"[{folder}] All trajectories filtered out. Check columns / data quality.")

    return traj_df, n_files_total


# -----------------------------
# Direction-sensitive scoring
# -----------------------------

def softplus(x: float) -> float:
    # numerically stable softplus-ish (for moderate values)
    if x > 50:
        return x
    if x < -50:
        return 0.0
    return float(np.log1p(np.exp(x)))


def score_candidate(
    delta_mean: float,
    delta_median: float,
    frac_mean_gt: float,
    frac_median_gt: float,
    w_delta_mean: float,
    w_delta_median: float,
    w_frac_mean: float,
    w_frac_median: float,
    neg_penalty: float,
    scale_delta: float
) -> float:
    """
    Direction-sensitive score:
    - Positive deltas contribute via softplus(delta/scale)
    - Negative deltas incur extra penalty (|delta|/scale)*neg_penalty
    - Fractions directly contribute.

    scale_delta controls how aggressively delta magnitudes are compressed.
    """
    dm = float(delta_mean)
    dmed = float(delta_median)

    pos_dm = softplus(dm / max(scale_delta, 1e-9))
    pos_dmed = softplus(dmed / max(scale_delta, 1e-9))

    neg_dm = max(0.0, -dm) / max(scale_delta, 1e-9)
    neg_dmed = max(0.0, -dmed) / max(scale_delta, 1e-9)

    s = 0.0
    s += w_delta_mean * pos_dm
    s += w_delta_median * pos_dmed
    s += w_frac_mean * float(frac_mean_gt)
    s += w_frac_median * float(frac_median_gt)

    # explicit penalty for violating prior
    s -= neg_penalty * (neg_dm + neg_dmed)
    return float(s)


# -----------------------------
# Sweep + selection
# -----------------------------

def make_gamma_grid(gamma_min: float, gamma_max: float, n_gamma: int, space: str) -> np.ndarray:
    if n_gamma <= 1:
        return np.array([gamma_min], dtype=float)
    if space == "linear":
        return np.linspace(gamma_min, gamma_max, n_gamma, dtype=float)
    # default: log space
    gmin = max(gamma_min, 1e-6)
    gmax = max(gamma_max, gmin * 1.000001)
    return np.exp(np.linspace(np.log(gmin), np.log(gmax), n_gamma, dtype=float))


def eval_one_eps(
    folder: str,
    eps_m: float,
    lat_col: str,
    lon_col: str,
    label_col: str,
    field_value: int,
    max_files: Optional[int],
    min_points: int,
    min_class_points_per_traj: int,
    max_points_per_traj: int,
    seed: int,
    logger: logging.Logger
) -> Dict[str, float]:
    traj_df, _ = load_traj_stats_for_eps(
        folder=folder,
        eps_m=eps_m,
        lat_col=lat_col,
        lon_col=lon_col,
        label_col=label_col,
        field_value=field_value,
        max_files=max_files,
        min_points=min_points,
        min_class_points_per_traj=min_class_points_per_traj,
        max_points_per_traj=max_points_per_traj,
        seed=seed,
        logger=logger
    )

    use = traj_df[traj_df["usable"] == True].copy()
    n_traj_used = int(len(use))
    if n_traj_used == 0:
        return {
            "n_traj_used": 0,
            "delta_mean": float("nan"),
            "delta_median": float("nan"),
            "frac_mean_gt": float("nan"),
            "frac_median_gt": float("nan"),
        }

    mean_diff = (use["mean_field"].to_numpy(dtype=float) - use["mean_road"].to_numpy(dtype=float))
    med_diff = (use["median_field"].to_numpy(dtype=float) - use["median_road"].to_numpy(dtype=float))

    delta_mean = float(np.nanmean(mean_diff))
    delta_median = float(np.nanmean(med_diff))

    frac_mean_gt = float(np.mean(mean_diff > 0))
    frac_median_gt = float(np.mean(med_diff > 0))

    return {
        "n_traj_used": n_traj_used,
        "delta_mean": delta_mean,
        "delta_median": delta_median,
        "frac_mean_gt": frac_mean_gt,
        "frac_median_gt": frac_median_gt,
    }


def plot_sweep(ds: str, sweep_df: pd.DataFrame, out_pdf: str) -> None:
    # No explicit colors; matplotlib default cycle is OK.
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(sweep_df["eps_m"], sweep_df["score"], marker="o", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\varepsilon$ (meters, log scale)")
    ax.set_ylabel("Score (direction-sensitive)")
    ax.grid(alpha=0.3)
    ax.set_title(f"{ds}: score vs eps")

    fig2, ax2 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax2.plot(sweep_df["eps_m"], sweep_df["delta_mean"], marker="o", linewidth=1, label="delta_mean")
    ax2.plot(sweep_df["eps_m"], sweep_df["delta_median"], marker="o", linewidth=1, label="delta_median")
    ax2.axhline(0.0, linewidth=1)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\varepsilon$ (meters, log scale)")
    ax2.set_ylabel("Delta (field - road)")
    ax2.grid(alpha=0.3)
    ax2.legend()
    ax2.set_title(f"{ds}: delta_mean / delta_median vs eps")

    fig3, ax3 = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax3.plot(sweep_df["eps_m"], sweep_df["frac_mean_gt"], marker="o", linewidth=1, label="frac_mean_gt")
    ax3.plot(sweep_df["eps_m"], sweep_df["frac_median_gt"], marker="o", linewidth=1, label="frac_median_gt")
    ax3.set_xscale("log")
    ax3.set_ylim(0.0, 1.05)
    ax3.set_xlabel(r"$\varepsilon$ (meters, log scale)")
    ax3.set_ylabel("Fraction of trajectories (field > road)")
    ax3.grid(alpha=0.3)
    ax3.legend()
    ax3.set_title(f"{ds}: fractions vs eps")

    # Save multipage PDF by simple trick: save 3 pages to same PDF via PdfPages
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig)
        pdf.savefig(fig2)
        pdf.savefig(fig3)

    plt.close(fig)
    plt.close(fig2)
    plt.close(fig3)


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", type=str, default="/home/ubuntu/forDataSet/AgriTraj",
                        help="Dataset root containing corn/wheat/paddy/tractor/harvest folders")
    parser.add_argument("--outdir", type=str, default="pic",
                        help="Output directory under script directory")
    parser.add_argument("--seed", type=int, default=42)

    # Base eps from ROAD knee (meters) — you can override if needed
    parser.add_argument("--wheat_eps_base", type=float, default=27.643851)
    parser.add_argument("--corn_eps_base", type=float, default=16.611668)
    parser.add_argument("--paddy_eps_base", type=float, default=159.602980)
    parser.add_argument("--tractor_eps_base", type=float, default=19.954843)
    parser.add_argument("--harvest_eps_base", type=float, default=8.814437)

    # Sweep range (WIDER)
    parser.add_argument("--gamma_min", type=float, default=0.5, help="Min gamma for eps = gamma*eps_base")
    parser.add_argument("--gamma_max", type=float, default=8.0, help="Max gamma for eps = gamma*eps_base")
    parser.add_argument("--n_gamma", type=int, default=45, help="Number of gamma grid points")
    parser.add_argument("--gamma_space", type=str, default="log", choices=["log", "linear"])

    # Optional absolute eps clamp (meters)
    parser.add_argument("--eps_abs_min", type=float, default=3.0, help="Clamp eps lower bound (m)")
    parser.add_argument("--eps_abs_max", type=float, default=2000.0, help="Clamp eps upper bound (m)")

    # Data & filtering
    parser.add_argument("--field_value", type=int, default=1,
                        help="Value indicating FIELD in label column; others treated as ROAD")
    parser.add_argument("--max_files_per_dataset", type=int, default=None,
                        help="Limit number of Excel files per dataset (debug only)")
    parser.add_argument("--min_points", type=int, default=5)
    parser.add_argument("--min_class_points_per_traj", type=int, default=5)
    parser.add_argument("--max_points_per_traj", type=int, default=0,
                        help="Downsample each trajectory to at most N points (0=disable). Useful for speed.")

    # Score weights (direction-sensitive)
    parser.add_argument("--w_delta_mean", type=float, default=0.6)
    parser.add_argument("--w_delta_median", type=float, default=1.0)
    parser.add_argument("--w_frac_mean", type=float, default=0.8)
    parser.add_argument("--w_frac_median", type=float, default=1.2)
    parser.add_argument("--neg_penalty", type=float, default=2.5,
                        help="Extra penalty multiplier when delta is negative (violates prior)")
    parser.add_argument("--scale_delta", type=float, default=50.0,
                        help="Scale to normalize delta magnitudes inside softplus/penalty")

    # ---- Explicit column names per dataset ----
    parser.add_argument("--corn_lat_col", type=str, default="latitude")
    parser.add_argument("--corn_lon_col", type=str, default="longitude")
    parser.add_argument("--corn_label_col", type=str, default="tags")

    parser.add_argument("--wheat_lat_col", type=str, default="纬度")
    parser.add_argument("--wheat_lon_col", type=str, default="经度")
    parser.add_argument("--wheat_label_col", type=str, default="标签")

    parser.add_argument("--paddy_lat_col", type=str, default="纬度")
    parser.add_argument("--paddy_lon_col", type=str, default="经度")
    parser.add_argument("--paddy_label_col", type=str, default="标记")

    parser.add_argument("--tractor_lat_col", type=str, default="latitude")
    parser.add_argument("--tractor_lon_col", type=str, default="longitude")
    parser.add_argument("--tractor_label_col", type=str, default="type")

    parser.add_argument("--harvest_lat_col", type=str, default="lat")
    parser.add_argument("--harvest_lon_col", type=str, default="long")
    parser.add_argument("--harvest_label_col", type=str, default="type")

    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(SCRIPT_DIR, args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, "eps_field_search.log")
    logger = setup_logger(log_path)

    logger.info("\n" + "=" * 90)
    logger.info("New run started at %s", datetime.datetime.now().isoformat())
    logger.info("Arguments: %s", vars(args))
    logger.info("=" * 90)

    os.chdir(args.root)

    ds_list = ["wheat", "corn", "paddy", "tractor", "harvest"]

    eps_base = {
        "wheat": float(args.wheat_eps_base),
        "corn": float(args.corn_eps_base),
        "paddy": float(args.paddy_eps_base),
        "tractor": float(args.tractor_eps_base),
        "harvest": float(args.harvest_eps_base),
    }

    colcfg = {
        "corn": (args.corn_lat_col, args.corn_lon_col, args.corn_label_col),
        "wheat": (args.wheat_lat_col, args.wheat_lon_col, args.wheat_label_col),
        "paddy": (args.paddy_lat_col, args.paddy_lon_col, args.paddy_label_col),
        "tractor": (args.tractor_lat_col, args.tractor_lon_col, args.tractor_label_col),
        "harvest": (args.harvest_lat_col, args.harvest_lon_col, args.harvest_label_col),
    }

    gamma_grid = make_gamma_grid(args.gamma_min, args.gamma_max, args.n_gamma, args.gamma_space)
    logger.info("Gamma grid: min=%.4f max=%.4f n=%d space=%s",
                float(gamma_grid.min()), float(gamma_grid.max()), int(gamma_grid.size), args.gamma_space)

    best_rows = []
    sweep_rows = []

    for ds in ds_list:
        lat_col, lon_col, lab_col = colcfg[ds]
        base = eps_base[ds]
        logger.info("\n--- Dataset: %s | eps_base=%.6f m | cols=(%s,%s,%s) ---", ds, base, lat_col, lon_col, lab_col)

        # sweep
        for g in tqdm(gamma_grid, desc=f"{ds} (eps sweep)", unit="γ"):
            eps_m = float(np.clip(g * base, args.eps_abs_min, args.eps_abs_max))

            metrics = eval_one_eps(
                folder=ds,
                eps_m=eps_m,
                lat_col=lat_col,
                lon_col=lon_col,
                label_col=lab_col,
                field_value=args.field_value,
                max_files=args.max_files_per_dataset,
                min_points=args.min_points,
                min_class_points_per_traj=args.min_class_points_per_traj,
                max_points_per_traj=args.max_points_per_traj,
                seed=args.seed,
                logger=logger
            )

            if metrics["n_traj_used"] <= 0 or (not np.isfinite(metrics["delta_mean"])) or (not np.isfinite(metrics["delta_median"])):
                score = float("nan")
            else:
                score = score_candidate(
                    delta_mean=metrics["delta_mean"],
                    delta_median=metrics["delta_median"],
                    frac_mean_gt=metrics["frac_mean_gt"],
                    frac_median_gt=metrics["frac_median_gt"],
                    w_delta_mean=args.w_delta_mean,
                    w_delta_median=args.w_delta_median,
                    w_frac_mean=args.w_frac_mean,
                    w_frac_median=args.w_frac_median,
                    neg_penalty=args.neg_penalty,
                    scale_delta=args.scale_delta
                )

            sweep_rows.append({
                "dataset": ds,
                "eps_base_from_road_knee_m": base,
                "gamma": float(g),
                "eps_m": eps_m,
                "eps_km": eps_m / 1000.0,
                "n_traj_used": int(metrics["n_traj_used"]),
                "delta_mean": float(metrics["delta_mean"]),
                "delta_median": float(metrics["delta_median"]),
                "frac_mean_gt": float(metrics["frac_mean_gt"]),
                "frac_median_gt": float(metrics["frac_median_gt"]),
                "score": float(score),
            })

        sweep_df = pd.DataFrame([r for r in sweep_rows if r["dataset"] == ds]).sort_values("eps_m")

        # select best by score (ignore NaNs)
        sweep_df2 = sweep_df[np.isfinite(sweep_df["score"])].copy()
        if len(sweep_df2) == 0:
            logger.warning("[%s] No valid sweep points produced. Skip selection.", ds)
            continue

        best_idx = int(sweep_df2["score"].to_numpy().argmax())
        best = sweep_df2.iloc[best_idx].to_dict()

        best_rows.append({
            "dataset": ds,
            "eps_base_from_road_knee_m": float(best["eps_base_from_road_knee_m"]),
            "eps_field_opt_m": float(best["eps_m"]),
            "eps_field_opt_km": float(best["eps_km"]),
            "gamma_best": float(best["gamma"]),
            "n_traj_used": int(best["n_traj_used"]),
            "delta_mean": float(best["delta_mean"]),
            "delta_median": float(best["delta_median"]),
            "frac_mean_gt": float(best["frac_mean_gt"]),
            "frac_median_gt": float(best["frac_median_gt"]),
            "score": float(best["score"]),
        })

        logger.info(
            "[%s] BEST eps=%.6f m (gamma=%.3f) | n_traj=%d | delta_mean=%.3f delta_median=%.3f | "
            "frac_mean_gt=%.3f frac_median_gt=%.3f | score=%.6f",
            ds,
            float(best["eps_m"]), float(best["gamma"]), int(best["n_traj_used"]),
            float(best["delta_mean"]), float(best["delta_median"]),
            float(best["frac_mean_gt"]), float(best["frac_median_gt"]),
            float(best["score"])
        )

        # plot sweep curves
        out_pdf = os.path.join(out_dir, f"eps_field_sweep_{ds}.pdf")
        plot_sweep(ds, sweep_df2.sort_values("eps_m"), out_pdf)
        logger.info("[%s] Sweep PDF saved -> %s", ds, out_pdf)

    # write outputs
    out_best = os.path.join(out_dir, "eps_field_opt.csv")
    out_sweep = os.path.join(out_dir, "eps_field_sweep.csv")

    df_best = pd.DataFrame(best_rows)
    df_sweep = pd.DataFrame(sweep_rows)

    # stable column order
    if len(df_best) > 0:
        df_best = df_best[[
            "dataset",
            "eps_base_from_road_knee_m",
            "eps_field_opt_m", "eps_field_opt_km",
            "gamma_best",
            "n_traj_used",
            "delta_mean", "delta_median",
            "frac_mean_gt", "frac_median_gt",
            "score"
        ]]

    if len(df_sweep) > 0:
        df_sweep = df_sweep[[
            "dataset",
            "eps_base_from_road_knee_m",
            "gamma",
            "eps_m", "eps_km",
            "n_traj_used",
            "delta_mean", "delta_median",
            "frac_mean_gt", "frac_median_gt",
            "score"
        ]]

    df_best.to_csv(out_best, index=False, encoding="utf-8-sig")
    df_sweep.to_csv(out_sweep, index=False, encoding="utf-8-sig")

    logger.info("\n=== Field-oriented EPS summary (written) ===")
    if len(df_best) > 0:
        logger.info(df_best.to_string(index=False))
    logger.info("Saved -> %s", out_best)
    logger.info("Saved -> %s", out_sweep)
    logger.info("Log   -> %s", log_path)

    print("\n=== Field-oriented EPS summary (written) ===")
    if len(df_best) > 0:
        print(df_best.to_string(index=False))
    else:
        print("(no best rows; check logs)")
    print(f"[OK] Saved -> {out_best}")
    print(f"[OK] Saved -> {out_sweep}")
    print(f"[OK] Log   -> {log_path}")


if __name__ == "__main__":
    main()
