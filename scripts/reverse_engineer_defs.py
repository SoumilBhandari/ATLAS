#!/usr/bin/env python3
"""
Reverse-engineer dataset PTP / Onset / Offset definitions (no docs required).

What it does:
1) Loads CSV
2) Detects float-named time columns (assumed time axis)
3) Sanity checks: whether dataset Onset/Offset lie in the same units/range as time axis
4) Grid-searches PTP window (start,end) that best matches dataset PTP (MAE / RMSE)
   - Uses RAW waveform (no filtering) to isolate "definition mismatch" first.
   - Optionally can run on baseline-corrected raw.

Usage:
  python3 reverse_engineer_defs.py /path/to/13.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


def is_floatlike(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def find_time_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if is_floatlike(str(c))]
    return sorted(cols, key=lambda c: float(c))


def time_axis_from_columns(time_cols: List[str]) -> np.ndarray:
    return np.asarray([float(c) for c in time_cols], dtype=np.float64)


def mask(time_ms: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (time_ms >= lo) & (time_ms <= hi)


def ptp_in_window(y: np.ndarray, m: np.ndarray) -> float:
    seg = y[m]
    if seg.size < 2:
        return np.nan
    return float(np.ptp(seg))


def run(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    time_cols = find_time_columns(df)
    if len(time_cols) < 10:
        raise RuntimeError("Not enough float-named time columns found.")

    t = time_axis_from_columns(time_cols)
    dt = float(np.median(np.diff(t)))
    print(f"Time axis: {len(t)} cols, range [{t[0]:.3f}, {t[-1]:.3f}] ms, dt≈{dt:.4f} ms, fs≈{1000/dt:.1f} Hz")

    # Pull raw waves
    waves = df[time_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)

    # Basic baseline correction (raw) option
    baseline_m = mask(t, -50.0, -10.0)
    baseline_mean = np.nanmean(waves[:, baseline_m], axis=1, keepdims=True)
    waves_bc = waves - baseline_mean

    # ---- Unit sanity checks for dataset columns ----
    for col in ["Onset", "Offset"]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
            if v.size:
                print(f"\nDataset {col}: N={v.size}  min={v.min():.3f}  median={np.median(v):.3f}  max={v.max():.3f}")
                in_range = np.mean((v >= t.min()) & (v <= t.max()))
                print(f"  fraction within time axis range: {in_range*100:.1f}%")
                # If very few are in-range, try alternative unit interpretations
                if in_range < 0.8:
                    # maybe seconds -> ms
                    v_ms = v * 1000.0
                    in_range2 = np.mean((v_ms >= t.min()) & (v_ms <= t.max()))
                    print(f"  if treated as seconds→ms: in-range {in_range2*100:.1f}%")
                    # maybe samples -> ms
                    v_samp_ms = v * dt
                    in_range3 = np.mean((v_samp_ms >= t.min()) & (v_samp_ms <= t.max()))
                    print(f"  if treated as samples×dt→ms: in-range {in_range3*100:.1f}%")

    # ---- Grid search PTP window ----
    if "PTP" not in df.columns:
        print("\nNo dataset 'PTP' column found; skipping PTP window reverse-engineer.")
        return

    ptp_ds = pd.to_numeric(df["PTP"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(ptp_ds) & np.isfinite(waves_bc).all(axis=1)
    print(f"\nPTP reverse-engineer: valid rows = {valid.sum()} / {len(df)}")
    if valid.sum() < 20:
        print("Not enough valid rows to do reliable window search.")
        return

    y = waves_bc[valid]
    ptp_ds = ptp_ds[valid]

    # Candidate window grid (ms)
    # Adjust this if your epoch is different.
    starts = np.arange(5.0, 31.0, 1.0)
    ends = np.arange(30.0, 91.0, 1.0)

    best = None  # (mae, rmse, start, end)
    for s in starts:
        for e in ends:
            if e <= s + 5:
                continue
            m = mask(t, s, e)
            if m.sum() < 5:
                continue
            ptp_ours = np.array([ptp_in_window(y[i], m) for i in range(y.shape[0])], dtype=np.float64)
            ok2 = np.isfinite(ptp_ours) & np.isfinite(ptp_ds)
            if ok2.sum() < 20:
                continue
            err = ptp_ours[ok2] - ptp_ds[ok2]
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
            if best is None or mae < best[0]:
                best = (mae, rmse, float(s), float(e))

    print("\nBest PTP window (using baseline-corrected RAW):")
    if best is None:
        print("  No valid window found.")
    else:
        mae, rmse, s, e = best
        print(f"  window = [{s:.1f}, {e:.1f}] ms   MAE={mae:.2f}   RMSE={rmse:.2f}")
        print("Interpretation: dataset PTP is most consistent with this post-stimulus window.")

    print("\nNext step if this still doesn’t match: dataset PTP may be computed on FILTERED signal or onset→offset instead of fixed window.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 reverse_engineer_defs.py /path/to/file.csv")
        sys.exit(1)
    run(Path(sys.argv[1]).expanduser().resolve())