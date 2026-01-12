#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset-level spatial coverage (Scheme A): bounding box (bbox) extent.
PLUS: within-dataset trajectory spatial overlap (Indicator A: grid redundancy)

What it does
------------
For each dataset folder (corn/wheat/paddy/tractor/harvest) under --root:
  1) Read all *.xls* files (each is one trajectory)
  2) Collect all valid lat/lon points across files
  3) Compute dataset-level bbox: min/max lat/lon
  4) Convert bbox spans to km and bbox area to km^2 (local equirectangular approximation)
  5) Compute overlap between each dataset bbox and the "train bbox" (union of training datasets):
       - train bbox is computed by merging points of datasets listed in --train_datasets
       - overlap area and overlap ratios are reported:
           * overlap / dataset_area
           * overlap / train_area
  6) Within-dataset trajectory overlap (Indicator A):
       - Discretize each trajectory into grid cells (meters).
       - For each trajectory i, obtain occupied cell set S_i.
       - Redundancy A = (sum_i |S_i|) / |union_i S_i|
         > 1 means trajectories overlap spatially; larger => more overlap/revisit.
  7) Multi-grid run:
       - Run Indicator A for multiple grid sizes (e.g., 50,100,150) in one execution.
       - Save per-grid CSV + per-grid bar chart + a combined CSV.

Outputs
-------
Under (script_dir / --outdir):
  - spatial_extent_bbox_bar.pdf
  - spatial_extent_bbox_stats.csv
  - spatial_overlap_with_train.csv
  - spatial_traj_overlap_A_g50.csv / ..._g100.csv / ..._g150.csv (per grid)
  - spatial_traj_overlap_A_bar_g50.pdf / ... (per grid)
  - spatial_traj_overlap_A_multi.csv (combined)
  - spatial_extent_overlap.log

Notes
-----
- bbox overlap is a conservative, coarse proxy for "spatial coverage overlap".
- Indicator A is deliberately simple & reviewer-friendly (grid occupancy, no GIS).
"""

import os
import glob
import argparse
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

EARTH_RADIUS_M = 6371008.8  # meters


# -----------------------------
# Geo helpers
# -----------------------------

def bbox_from_points(lat: np.ndarray, lon: np.ndarray) -> Dict[str, float]:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat = lat[np.isfinite(lat)]
    lon = lon[np.isfinite(lon)]
    if lat.size == 0 or lon.size == 0:
        return {
            "min_lat": np.nan, "max_lat": np.nan,
            "min_lon": np.nan, "max_lon": np.nan,
        }
    return {
        "min_lat": float(np.min(lat)),
        "max_lat": float(np.max(lat)),
        "min_lon": float(np.min(lon)),
        "max_lon": float(np.max(lon)),
    }


def bbox_spans_km_and_area_km2(b: Dict[str, float]) -> Tuple[float, float, float]:
    """
    Convert bbox to:
      - lat_span_km
      - lon_span_km (at mid-lat)
      - area_km2  ~ (lat_span_km * lon_span_km)
    """
    if (not np.isfinite(b["min_lat"]) or not np.isfinite(b["max_lat"]) or
        not np.isfinite(b["min_lon"]) or not np.isfinite(b["max_lon"])):
        return np.nan, np.nan, np.nan

    min_lat, max_lat = b["min_lat"], b["max_lat"]
    min_lon, max_lon = b["min_lon"], b["max_lon"]

    dlat = np.deg2rad(max_lat - min_lat)
    dlon = np.deg2rad(max_lon - min_lon)
    mid_lat = np.deg2rad(0.5 * (min_lat + max_lat))

    lat_span_m = EARTH_RADIUS_M * dlat
    lon_span_m = EARTH_RADIUS_M * np.cos(mid_lat) * dlon

    lat_span_km = float(lat_span_m / 1000.0)
    lon_span_km = float(lon_span_m / 1000.0)
    area_km2 = float((lat_span_m * lon_span_m) / 1e6)

    lat_span_km = abs(lat_span_km)
    lon_span_km = abs(lon_span_km)
    area_km2 = abs(area_km2)
    return lat_span_km, lon_span_km, area_km2


def bbox_intersection_area_km2(b1: Dict[str, float], b2: Dict[str, float]) -> float:
    """Intersection area of two bboxes, approximated in km^2."""
    keys = ["min_lat", "max_lat", "min_lon", "max_lon"]
    if any(not np.isfinite(b1[k]) for k in keys) or any(not np.isfinite(b2[k]) for k in keys):
        return np.nan

    min_lat = max(b1["min_lat"], b2["min_lat"])
    max_lat = min(b1["max_lat"], b2["max_lat"])
    min_lon = max(b1["min_lon"], b2["min_lon"])
    max_lon = min(b1["max_lon"], b2["max_lon"])

    if (max_lat <= min_lat) or (max_lon <= min_lon):
        return 0.0

    b_int = {"min_lat": min_lat, "max_lat": max_lat, "min_lon": min_lon, "max_lon": max_lon}
    _, _, area_km2 = bbox_spans_km_and_area_km2(b_int)
    return float(area_km2)


def latlon_to_xy_m(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float, lon0_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local equirectangular approximation:
      x = R cos(lat0) (lon-lon0)
      y = R (lat-lat0)
    Inputs in degrees, outputs in meters.
    """
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    lat0 = np.deg2rad(float(lat0_deg))
    lon0 = np.deg2rad(float(lon0_deg))

    x = EARTH_RADIUS_M * np.cos(lat0) * (lon - lon0)
    y = EARTH_RADIUS_M * (lat - lat0)
    return x, y


# -----------------------------
# Data loading (dataset-level points)
# -----------------------------

def collect_latlon_from_folder(
    folder: str,
    lat_col: str,
    lon_col: str,
    max_files: Optional[int],
    logger: logging.Logger,
    min_points_per_file: int = 5,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Returns:
      lat_all, lon_all, n_files_total, n_files_used
    """
    files = sorted(glob.glob(os.path.join(folder, "*.xls*")))
    n_total = len(files)
    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        logger.warning(f"[{folder}] No Excel files found.")
        return np.array([], dtype=float), np.array([], dtype=float), 0, 0

    lat_list, lon_list = [], []
    n_used = 0

    for fp in tqdm(files, desc=f"{folder}", unit="file"):
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            logger.warning(f"[{folder}] failed to read {fp}: {e}")
            continue

        if lat_col not in df.columns or lon_col not in df.columns:
            logger.warning(
                f"[{folder}] {os.path.basename(fp)} missing lat/lon columns "
                f"({lat_col},{lon_col}). Available={df.columns.tolist()}"
            )
            continue

        lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(lat) & np.isfinite(lon)
        lat, lon = lat[m], lon[m]

        if lat.size < min_points_per_file:
            continue

        lat_list.append(lat)
        lon_list.append(lon)
        n_used += 1

    if n_used == 0:
        logger.warning(f"[{folder}] All files filtered out (no usable lat/lon).")
        return np.array([], dtype=float), np.array([], dtype=float), n_total, 0

    lat_all = np.concatenate(lat_list)
    lon_all = np.concatenate(lon_list)
    return lat_all, lon_all, n_total, n_used


# -----------------------------
# Trajectory overlap (Indicator A)
# -----------------------------

def trajectory_grid_cells(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float,
    lon0: float,
    grid_size_m: float
) -> Set[Tuple[int, int]]:
    """
    Convert one trajectory lat/lon to occupied grid cell indices (cx, cy).
    """
    x, y = latlon_to_xy_m(lat, lon, lat0, lon0)
    cx = np.floor(x / float(grid_size_m)).astype(int)
    cy = np.floor(y / float(grid_size_m)).astype(int)
    return set(zip(cx.tolist(), cy.tolist()))


def compute_dataset_redundancy_A(
    folder: str,
    lat_col: str,
    lon_col: str,
    max_files: Optional[int],
    logger: logging.Logger,
    grid_size_m: float,
    min_points_per_file: int = 5,
) -> Dict[str, float]:
    """
    Indicator A: Redundancy = (sum_i |S_i|) / |union_i S_i|
    where S_i is the occupied grid cell set of trajectory i.
    """
    files = sorted(glob.glob(os.path.join(folder, "*.xls*")))
    n_total = len(files)
    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        return {
            "dataset": folder,
            "n_files_total": 0,
            "n_traj_used": 0,
            "grid_size_m": float(grid_size_m),
            "sum_cells": np.nan,
            "unique_cells": np.nan,
            "redundancy_A": np.nan,
        }

    # First pass: robust reference origin (median of all points)
    lat_all, lon_all, _, _ = collect_latlon_from_folder(
        folder=folder,
        lat_col=lat_col,
        lon_col=lon_col,
        max_files=max_files,
        logger=logger,
        min_points_per_file=min_points_per_file
    )
    if lat_all.size == 0 or lon_all.size == 0:
        return {
            "dataset": folder,
            "n_files_total": int(n_total),
            "n_traj_used": 0,
            "grid_size_m": float(grid_size_m),
            "sum_cells": np.nan,
            "unique_cells": np.nan,
            "redundancy_A": np.nan,
        }

    lat0 = float(np.median(lat_all))
    lon0 = float(np.median(lon_all))

    union_cells: Set[Tuple[int, int]] = set()
    sum_cells = 0
    n_traj_used = 0

    for fp in tqdm(files, desc=f"{folder}-overlapA-{int(grid_size_m)}m", unit="file"):
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            logger.warning(f"[{folder}][A] failed to read {fp}: {e}")
            continue

        if lat_col not in df.columns or lon_col not in df.columns:
            continue

        lat = pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(df[lon_col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(lat) & np.isfinite(lon)
        lat, lon = lat[m], lon[m]

        if lat.size < min_points_per_file:
            continue

        cells = trajectory_grid_cells(lat, lon, lat0=lat0, lon0=lon0, grid_size_m=grid_size_m)
        if len(cells) == 0:
            continue

        sum_cells += len(cells)
        union_cells |= cells
        n_traj_used += 1

    unique_cells = len(union_cells)
    redundancy = float(sum_cells / (unique_cells + 1e-12)) if unique_cells > 0 else float("nan")

    return {
        "dataset": folder,
        "n_files_total": int(n_total),
        "n_traj_used": int(n_traj_used),
        "grid_size_m": float(grid_size_m),
        "sum_cells": int(sum_cells),
        "unique_cells": int(unique_cells),
        "redundancy_A": redundancy,
        "lat0_median": lat0,
        "lon0_median": lon0,
    }


# -----------------------------
# Plotting
# -----------------------------

def plot_bbox_area_bar(ds_list: List[str], area_km2: List[float], out_path: str, title: str):
    x = np.arange(len(ds_list))
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar(x, area_km2)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_list)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("BBox spatial extent (km$^2$)")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13)
    for i, v in enumerate(area_km2):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

def plot_redundancy_bar(ds_list: List[str], vals: List[float], out_path: str, title: str):
    x = np.arange(len(ds_list))
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.bar(x, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(ds_list)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Within-dataset overlap (Redundancy A)")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title, fontsize=13)

    # ---- NEW: y-axis starts from 1 (theoretical minimum) ----
    finite_vals = [v for v in vals if np.isfinite(v)]
    if len(finite_vals) > 0:
        vmin = 1.0
        vmax = float(max(finite_vals))
        # add a small headroom for text annotations
        headroom = max(0.05, 0.08 * (vmax - vmin + 1e-12))
        ax.set_ylim(vmin, vmax + headroom)
    else:
        ax.set_ylim(1.0, 2.0)  # safe fallback

    # annotate
    for i, v in enumerate(vals):
        if np.isfinite(v):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

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

    parser.add_argument("--max_files_per_dataset", type=int, default=None,
                        help="Optional cap on files per dataset for quick checks")

    parser.add_argument("--train_datasets", type=str, default="corn,wheat,paddy,tractor",
                        help="Comma-separated dataset folder names to form the training bbox")

    # Backward-compatible single grid size
    parser.add_argument("--grid_size_m", type=float, default=100.0,
                        help="Single grid size (meters) for Indicator A (kept for backward compatibility).")

    # NEW: multi-grid run
    parser.add_argument("--grid_sizes_m", type=str, default=None,
                        help="Comma-separated grid sizes (meters) for multi-run, e.g., 50,100,150. "
                             "If set, overrides --grid_size_m.")

    # Explicit column names per dataset
    parser.add_argument("--corn_lat_col", type=str, default="latitude")
    parser.add_argument("--corn_lon_col", type=str, default="longitude")

    parser.add_argument("--wheat_lat_col", type=str, default="纬度")
    parser.add_argument("--wheat_lon_col", type=str, default="经度")

    parser.add_argument("--paddy_lat_col", type=str, default="纬度")
    parser.add_argument("--paddy_lon_col", type=str, default="经度")

    parser.add_argument("--tractor_lat_col", type=str, default="latitude")
    parser.add_argument("--tractor_lon_col", type=str, default="longitude")

    parser.add_argument("--harvest_lat_col", type=str, default="lat")
    parser.add_argument("--harvest_lon_col", type=str, default="long")

    args = parser.parse_args()

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(SCRIPT_DIR, args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    # Outputs (bbox)
    out_pdf = os.path.join(out_dir, "spatial_extent_bbox_bar.pdf")
    out_csv = os.path.join(out_dir, "spatial_extent_bbox_stats.csv")
    out_overlap_csv = os.path.join(out_dir, "spatial_overlap_with_train.csv")

    # Outputs (Indicator A multi)
    out_overlapA_multi_csv = os.path.join(out_dir, "spatial_traj_overlap_A_multi.csv")

    log_path = os.path.join(out_dir, "spatial_extent_overlap.log")

    logging.basicConfig(
        filename=log_path,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger("SpatialExtentOverlap")

    logger.info("\n" + "=" * 90)
    logger.info("New run started at %s", datetime.datetime.now().isoformat())
    logger.info("Arguments: %s", vars(args))
    logger.info("Script dir: %s", SCRIPT_DIR)
    logger.info("Output dir: %s", out_dir)
    logger.info("=" * 90)

    ds_list = ["corn", "wheat", "paddy", "tractor", "harvest"]
    colcfg = {
        "corn": (args.corn_lat_col, args.corn_lon_col),
        "wheat": (args.wheat_lat_col, args.wheat_lon_col),
        "paddy": (args.paddy_lat_col, args.paddy_lon_col),
        "tractor": (args.tractor_lat_col, args.tractor_lon_col),
        "harvest": (args.harvest_lat_col, args.harvest_lon_col),
    }

    os.chdir(args.root)

    # 1) Per-dataset bbox stats
    rows: List[Dict] = []
    ds_bbox: Dict[str, Dict[str, float]] = {}
    ds_area: Dict[str, float] = {}
    ds_points: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for name in ds_list:
        lat_col, lon_col = colcfg[name]
        logger.info(f"Processing dataset: {name} | cols=({lat_col},{lon_col})")

        lat_all, lon_all, n_total, n_used = collect_latlon_from_folder(
            folder=name,
            lat_col=lat_col,
            lon_col=lon_col,
            max_files=args.max_files_per_dataset,
            logger=logger,
            min_points_per_file=5
        )
        ds_points[name] = (lat_all, lon_all)

        b = bbox_from_points(lat_all, lon_all)
        lat_span_km, lon_span_km, area_km2 = bbox_spans_km_and_area_km2(b)

        ds_bbox[name] = b
        ds_area[name] = area_km2

        row = {
            "dataset": name,
            "n_files_total": int(n_total),
            "n_files_used": int(n_used),
            "n_points_used": int(lat_all.size),

            "min_lat": b["min_lat"],
            "max_lat": b["max_lat"],
            "min_lon": b["min_lon"],
            "max_lon": b["max_lon"],

            "lat_span_km": lat_span_km,
            "lon_span_km": lon_span_km,
            "bbox_area_km2": area_km2,
        }
        rows.append(row)

        logger.info(
            f"[{name}] files_used={n_used}/{n_total}, points={lat_all.size} | "
            f"bbox: lat[{b['min_lat']:.6f},{b['max_lat']:.6f}] lon[{b['min_lon']:.6f},{b['max_lon']:.6f}] | "
            f"span_km(lat={lat_span_km:.3f}, lon={lon_span_km:.3f}) area_km2={area_km2:.3f}"
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved bbox stats CSV -> {out_csv}")

    # 2) Train bbox (merged points of training datasets)
    train_names = [s.strip() for s in args.train_datasets.split(",") if s.strip()]
    train_lat_list, train_lon_list = [], []
    for tn in train_names:
        if tn not in ds_points:
            logger.warning(f"Train dataset '{tn}' not in ds_list. Skipped.")
            continue
        lat_all, lon_all = ds_points[tn]
        if lat_all.size > 0:
            train_lat_list.append(lat_all)
            train_lon_list.append(lon_all)

    if len(train_lat_list) == 0:
        raise RuntimeError("No training points collected. Check --train_datasets and column names.")

    train_lat = np.concatenate(train_lat_list)
    train_lon = np.concatenate(train_lon_list)
    train_bbox = bbox_from_points(train_lat, train_lon)
    _, _, train_area_km2 = bbox_spans_km_and_area_km2(train_bbox)

    logger.info(
        f"[TRAIN_BBOX] datasets={train_names} | "
        f"bbox: lat[{train_bbox['min_lat']:.6f},{train_bbox['max_lat']:.6f}] "
        f"lon[{train_bbox['min_lon']:.6f},{train_bbox['max_lon']:.6f}] | "
        f"area_km2={train_area_km2:.3f}"
    )

    # 3) Overlap with train bbox
    overlap_rows: List[Dict] = []
    for name in ds_list:
        b = ds_bbox[name]
        area = ds_area[name]
        ov_area = bbox_intersection_area_km2(b, train_bbox)

        ov_over_ds = float(ov_area / (area + 1e-12)) if np.isfinite(area) else np.nan
        ov_over_train = float(ov_area / (train_area_km2 + 1e-12)) if np.isfinite(train_area_km2) else np.nan

        overlap_rows.append({
            "dataset": name,
            "dataset_area_km2": area,
            "train_area_km2": train_area_km2,
            "overlap_area_km2": ov_area,
            "overlap_over_dataset": ov_over_ds,
            "overlap_over_train": ov_over_train,
        })

        logger.info(
            f"[OVERLAP] {name} vs TRAIN | overlap_km2={ov_area:.3f} | "
            f"overlap/dataset={ov_over_ds:.3f} | overlap/train={ov_over_train:.3f}"
        )

    pd.DataFrame(overlap_rows).to_csv(out_overlap_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved overlap CSV -> {out_overlap_csv}")

    # 4) Plot bbox area bar chart
    area_list = [ds_area[n] for n in ds_list]
    plot_bbox_area_bar(
        ds_list=ds_list,
        area_km2=area_list,
        out_path=out_pdf,
        title="Dataset-level spatial extent (bounding-box area)"
    )
    logger.info(f"Saved bbox area bar chart -> {out_pdf}")

    # 5) Indicator A multi-grid run
    if args.grid_sizes_m is not None and str(args.grid_sizes_m).strip() != "":
        grid_list = [float(x.strip()) for x in args.grid_sizes_m.split(",") if x.strip()]
    else:
        grid_list = [float(args.grid_size_m)]

    logger.info(f"=== Indicator A (Grid Redundancy) | grid_sizes_m={grid_list} ===")

    overlapA_multi_rows: List[Dict] = []

    for g in grid_list:
        per_grid_rows: List[Dict] = []
        vals: List[float] = []

        for name in ds_list:
            lat_col, lon_col = colcfg[name]
            resA = compute_dataset_redundancy_A(
                folder=name,
                lat_col=lat_col,
                lon_col=lon_col,
                max_files=args.max_files_per_dataset,
                logger=logger,
                grid_size_m=g,
                min_points_per_file=5
            )
            per_grid_rows.append(resA)
            overlapA_multi_rows.append(resA)
            vals.append(float(resA["redundancy_A"]) if np.isfinite(resA["redundancy_A"]) else np.nan)

            logger.info(
                f"[A][grid={g:.0f}m][{name}] n_traj_used={resA.get('n_traj_used', 0)} | "
                f"sum_cells={resA.get('sum_cells', np.nan)} | unique_cells={resA.get('unique_cells', np.nan)} | "
                f"redundancy_A={resA.get('redundancy_A', np.nan):.4f}"
            )

        # per-grid CSV
        out_csv_g = os.path.join(out_dir, f"spatial_traj_overlap_A_g{int(g)}.csv")
        pd.DataFrame(per_grid_rows).to_csv(out_csv_g, index=False, encoding="utf-8-sig")
        logger.info(f"Saved Indicator-A CSV (grid={g:.0f}m) -> {out_csv_g}")

        # per-grid bar plot
        out_pdf_g = os.path.join(out_dir, f"spatial_traj_overlap_A_bar_g{int(g)}.pdf")
        plot_redundancy_bar(
            ds_list=ds_list,
            vals=vals,
            out_path=out_pdf_g,
            title=f"Within-dataset trajectory overlap (Redundancy A, grid={g:.0f}m)"
        )
        logger.info(f"Saved Indicator-A bar chart (grid={g:.0f}m) -> {out_pdf_g}")

    # combined multi-grid CSV
    pd.DataFrame(overlapA_multi_rows).to_csv(out_overlapA_multi_csv, index=False, encoding="utf-8-sig")
    logger.info(f"Saved Indicator-A multi-grid CSV -> {out_overlapA_multi_csv}")

    print(f"Saved bbox area bar chart              -> {out_pdf}")
    print(f"Saved bbox stats CSV                   -> {out_csv}")
    print(f"Saved bbox overlap-with-train CSV      -> {out_overlap_csv}")
    print(f"Saved Indicator-A multi-grid CSV       -> {out_overlapA_multi_csv}")
    print(f"Log written to                         -> {log_path}")


if __name__ == "__main__":
    main()
