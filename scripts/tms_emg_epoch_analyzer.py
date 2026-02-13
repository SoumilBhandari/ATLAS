#!/usr/bin/env python3
"""
TMS-EMG epoch analyzer (AUTO reverse-engineer edition v2)
---------------------------------------------------------
Automatically infers what dataset columns (PTP/Onset/Offset) mean by
grid-searching signal-processing definitions against ground truth.

v2 fixes:
  - Coverage-penalised scoring: score = MAE_on + MAE_off + λ*(1 - coverage)
    so the optimizer can't cheat by detecting only an easy subset.
  - Two onset families searched:
    a) Forward-threshold: first sustained envelope > threshold (hard min latency)
    b) Peak-anchored onset: find envelope peak, walk backward to where it crossed threshold
  - Offset remains peak-anchored (forward from peak + gap)

Usage:
    python tms_emg_epoch_analyzer_auto.py data.csv -v
    python tms_emg_epoch_analyzer_auto.py data.csv --coverage-lambda 40
    python tms_emg_epoch_analyzer_auto.py data.csv --no-plots --max-fit-trials 300
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════
def find_time_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        try:
            float(c)
            cols.append(c)
        except (ValueError, TypeError):
            pass
    return sorted(cols, key=lambda c: float(c))


def time_axis_from_columns(time_cols: Sequence[str]) -> np.ndarray:
    return np.array([float(c) for c in time_cols], dtype=np.float64)


def infer_fs_hz(time_ms: np.ndarray) -> float:
    if time_ms.size < 2:
        return 5000.0
    dt = float(np.median(np.diff(time_ms)))
    return 1000.0 / dt if dt > 0 else 5000.0


def _mask(time_ms: np.ndarray, window: Tuple[float, float]) -> np.ndarray:
    return (time_ms >= window[0]) & (time_ms <= window[1])


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if a.size else float("inf")


def _consecutive_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if mask.size == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    padded = np.empty(mask.size + 2, dtype=np.int8)
    padded[0] = 0
    padded[-1] = 0
    padded[1:-1] = mask
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return starts, ends - starts


def _first_run(
    global_idx: np.ndarray, local_mask: np.ndarray, min_len: int,
) -> Optional[int]:
    starts, lengths = _consecutive_runs(local_mask)
    if starts.size == 0:
        return None
    good = np.flatnonzero(lengths >= min_len)
    if good.size == 0:
        return None
    return int(global_idx[starts[good[0]]])


# ═══════════════════════════════════════════════════════════════════
#  Signal processing
# ═══════════════════════════════════════════════════════════════════
def blank_artifact(
    time_ms: np.ndarray, y: np.ndarray,
    window: Tuple[float, float], method: str = "linear",
) -> np.ndarray:
    out = y.copy()
    m = _mask(time_ms, window)
    if not np.any(m):
        return out
    idx = np.flatnonzero(m)
    i0, i1 = idx[0], idx[-1]
    left = max(i0 - 1, 0)
    right = min(i1 + 1, len(y) - 1)
    if method == "zero":
        out[idx] = 0.0
    else:
        out[idx] = np.linspace(out[left], out[right], len(idx))
    return out


def baseline_correct(
    time_ms: np.ndarray, y: np.ndarray, window: Tuple[float, float],
) -> np.ndarray:
    m = _mask(time_ms, window)
    if not np.any(m):
        return y.copy()
    return y - float(np.mean(y[m]))


def build_filters(
    fs: float, bandpass_hz: Tuple[float, float], bandpass_order: int,
    notch_hz: float, notch_q: float, notch_harmonics: Tuple[int, ...],
) -> Tuple[Optional[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]:
    bp_sos = None
    nyq = fs / 2.0
    lo_n = max(0.1, bandpass_hz[0]) / nyq
    hi_n = min(bandpass_hz[1], nyq - 1.0) / nyq
    if 0 < lo_n < hi_n < 1:
        bp_sos = butter(bandpass_order, [lo_n, hi_n], btype="bandpass", output="sos")
    else:
        logger.warning("Bandpass invalid for fs=%.1f Hz. Skipping.", fs)

    notch_bas: List[Tuple[np.ndarray, np.ndarray]] = []
    for harmonic in (1, *notch_harmonics):
        freq = notch_hz * harmonic
        if freq < nyq:
            b, a = iirnotch(w0=freq, Q=notch_q, fs=fs)
            notch_bas.append((b, a))

    return bp_sos, notch_bas


def apply_filters(
    y: np.ndarray, bp_sos: Optional[np.ndarray],
    notch_bas: List[Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    out = y.copy()
    if bp_sos is not None:
        out = sosfiltfilt(bp_sos, out)
    for b, a in notch_bas:
        out = filtfilt(b, a, out)
    return out


def rms_envelope(y: np.ndarray, win_n: int) -> np.ndarray:
    win_n = max(1, win_n)
    n = y.size
    cs2 = np.empty(n + 1, dtype=np.float64)
    cs2[0] = 0.0
    np.cumsum(y * y, out=cs2[1:])
    idx = np.arange(n)
    lo = np.maximum(idx + 1 - win_n, 0)
    counts = (idx + 1 - lo).astype(np.float64)
    return np.sqrt((cs2[idx + 1] - cs2[lo]) / counts)


def rms_envelope_batch(signals: np.ndarray, win_n: int) -> np.ndarray:
    win_n = max(1, win_n)
    n_trials, n_t = signals.shape
    cs2 = np.empty((n_trials, n_t + 1), dtype=np.float64)
    cs2[:, 0] = 0.0
    np.cumsum(signals * signals, axis=1, out=cs2[:, 1:])
    idx = np.arange(n_t)
    lo = np.maximum(idx + 1 - win_n, 0)
    counts = (idx + 1 - lo).astype(np.float64)
    return np.sqrt((cs2[:, idx + 1] - cs2[:, lo]) / counts)


# ═══════════════════════════════════════════════════════════════════
#  Config dataclasses
# ═══════════════════════════════════════════════════════════════════
def _default_notch_harmonics() -> Tuple[int, ...]:
    return (2, 3)


@dataclass(frozen=True)
class PreprocConfig:
    baseline_window_ms: Tuple[float, float] = (-50.0, -10.0)
    blank_window_ms: Tuple[float, float] = (-1.0, 6.0)
    blank_method: str = "linear"
    do_filter: bool = True
    bandpass_hz: Tuple[float, float] = (20.0, 450.0)
    bandpass_order: int = 4
    notch_hz: float = 60.0
    notch_q: float = 30.0
    notch_harmonics: Tuple[int, ...] = field(default_factory=_default_notch_harmonics)


@dataclass(frozen=True)
class PTPDef:
    use_signal: str
    window_ms: Tuple[float, float]


@dataclass(frozen=True)
class OnOffDef:
    use_signal: str          # "raw_bc" or "filt_bc"
    onset_method: str        # "forward" or "peak_anchored"
    smooth_ms: float
    onset_k: float           # threshold for forward onset
    offset_k: float
    min_above_ms: float      # sustain for forward onset
    onset_backtrack_k: float # threshold for peak-anchored backward search
    hold_ms: float           # sustain for offset below-threshold
    gap_ms: float            # gap after peak before offset search
    onset_search_ms: Tuple[float, float]
    offset_search_end_ms: float
    min_onset_ms: float
    peak_search_ms: Tuple[float, float]  # window to find envelope peak


# ═══════════════════════════════════════════════════════════════════
#  Preprocessing (batch)
# ═══════════════════════════════════════════════════════════════════
def preprocess_all(
    df: pd.DataFrame, time_cols: List[str], time_ms: np.ndarray,
    fs: float, pcfg: PreprocConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (raw_bc, filt_bc), each shape (n_trials, n_time)."""
    waves = df[time_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    n_trials = waves.shape[0]

    bp_sos, notch_bas = (None, [])
    if pcfg.do_filter:
        bp_sos, notch_bas = build_filters(
            fs, pcfg.bandpass_hz, pcfg.bandpass_order,
            pcfg.notch_hz, pcfg.notch_q, pcfg.notch_harmonics,
        )

    raw_bc = np.empty_like(waves)
    filt_bc = np.empty_like(waves)

    for i in range(n_trials):
        y = waves[i]
        nan_frac = float(np.mean(np.isnan(y)))
        if nan_frac > 0.1:
            raw_bc[i] = np.nan
            filt_bc[i] = np.nan
            continue
        if nan_frac > 0:
            y = pd.Series(y).interpolate(limit_direction="both").to_numpy()

        yb = blank_artifact(time_ms, y, pcfg.blank_window_ms, pcfg.blank_method)
        raw_bc[i] = baseline_correct(time_ms, yb, pcfg.baseline_window_ms)
        yf = apply_filters(yb, bp_sos, notch_bas) if pcfg.do_filter else yb
        filt_bc[i] = baseline_correct(time_ms, yf, pcfg.baseline_window_ms)

    return raw_bc, filt_bc


# ═══════════════════════════════════════════════════════════════════
#  PTP inference (solved — unchanged from v1)
# ═══════════════════════════════════════════════════════════════════
def infer_ptp_definition(
    df: pd.DataFrame, time_ms: np.ndarray,
    raw_bc: np.ndarray, filt_bc: np.ndarray,
    max_trials: int = 250,
) -> Optional[PTPDef]:
    if "PTP" not in df.columns:
        return None

    ds_ptp = pd.to_numeric(df["PTP"], errors="coerce").to_numpy()
    ok = np.isfinite(ds_ptp)
    if ok.sum() < 50:
        logger.warning("Not enough valid PTP values (%d) to infer.", ok.sum())
        return None

    rng = np.random.default_rng(0)
    idx = np.flatnonzero(ok)
    if idx.size > max_trials:
        idx = rng.choice(idx, size=max_trials, replace=False)
    ds = ds_ptp[idx]

    dt = float(np.median(np.diff(time_ms)))
    starts = np.arange(0.0, 60.0 + 1e-9, dt)
    lengths = np.arange(2.0, 25.0 + 1e-9, dt)

    def _search(sig: np.ndarray) -> Tuple[float, Tuple[float, float]]:
        sub = sig[idx]
        best_mae = float("inf")
        best_win = (0.0, 0.0)
        for s in starts:
            for L in lengths:
                e = s + L
                m = _mask(time_ms, (float(s), float(e)))
                if not np.any(m):
                    continue
                pred = np.ptp(sub[:, m], axis=1)
                mae = float(np.mean(np.abs(pred - ds)))
                if mae < best_mae:
                    best_mae = mae
                    best_win = (float(s), float(e))
        return best_mae, best_win

    t0 = _time.monotonic()
    mae_raw, win_raw = _search(raw_bc)
    mae_filt, win_filt = _search(filt_bc)
    elapsed = _time.monotonic() - t0

    if mae_raw <= mae_filt:
        logger.info("PTP inferred (%.1fs): raw_bc [%.2f, %.2f] ms  MAE=%.3f",
                     elapsed, win_raw[0], win_raw[1], mae_raw)
        return PTPDef(use_signal="raw_bc", window_ms=win_raw)
    else:
        logger.info("PTP inferred (%.1fs): filt_bc [%.2f, %.2f] ms  MAE=%.3f",
                     elapsed, win_filt[0], win_filt[1], mae_filt)
        return PTPDef(use_signal="filt_bc", window_ms=win_filt)


# ═══════════════════════════════════════════════════════════════════
#  Onset / Offset detection — two onset families
# ═══════════════════════════════════════════════════════════════════
_BASELINE_WINDOW = (-50.0, -10.0)


def _env_baseline_stats(
    env: np.ndarray, time_ms: np.ndarray,
) -> Tuple[float, float]:
    bl = _mask(time_ms, _BASELINE_WINDOW)
    if not np.any(bl):
        return 0.0, 0.0
    bl_env = env[bl]
    mu = float(np.mean(bl_env))
    sd = float(np.std(bl_env, ddof=1)) if bl_env.size > 1 else 0.0
    return mu, sd


def _detect_onset_forward(
    env: np.ndarray, time_ms: np.ndarray,
    thr: float, dt: float, d: OnOffDef,
) -> Optional[float]:
    """Forward onset: first sustained env > thr within onset_search window."""
    si = np.flatnonzero(_mask(time_ms, d.onset_search_ms))
    if si.size == 0:
        return None
    min_above_n = max(1, round(d.min_above_ms / dt))
    onset_i = _first_run(si, env[si] > thr, min_above_n)
    if onset_i is None:
        return None
    t = float(time_ms[onset_i])
    return t if t >= d.min_onset_ms else None


def _detect_onset_peak_anchored(
    env: np.ndarray, time_ms: np.ndarray,
    thr: float, dt: float, d: OnOffDef,
) -> Optional[float]:
    """
    Peak-anchored onset:
    1. Find envelope peak in peak_search window.
    2. Search backward from peak: find the last time envelope crossed UP
       through threshold before the peak. That crossing is the onset.
    Enforces min_onset_ms by starting the backward search from there.
    """
    peak_idx = np.flatnonzero(_mask(time_ms, d.peak_search_ms))
    if peak_idx.size == 0:
        return None

    peak_i = peak_idx[np.argmax(env[peak_idx])]

    # Peak must be a real response (above threshold)
    if env[peak_i] <= thr:
        return None

    # Search from max(onset_search_start, min_onset_ms) up to peak
    effective_start = max(d.onset_search_ms[0], d.min_onset_ms)
    search_start_i = max(0, np.searchsorted(time_ms, effective_start))
    if search_start_i >= peak_i:
        return None

    # Segment from effective start to peak (inclusive)
    seg = env[search_start_i:peak_i + 1]
    below = seg <= thr

    if not np.any(below):
        # Entire segment above threshold — onset is at effective_start
        return float(time_ms[search_start_i])

    # Find the LAST below-threshold sample, then onset = next sample after it
    last_below_local = int(np.flatnonzero(below)[-1])
    onset_local = last_below_local + 1
    if onset_local >= len(seg):
        return None

    onset_i = search_start_i + onset_local

    # Sub-sample interpolation: linearly interpolate the crossing time
    # between the last below-threshold sample and the first above-threshold sample
    below_i = onset_i - 1
    if below_i >= 0 and env[below_i] < thr < env[onset_i]:
        denom = env[onset_i] - env[below_i]
        if denom > 1e-12:
            frac = (thr - env[below_i]) / denom
            return float(time_ms[below_i] + frac * dt)

    return float(time_ms[onset_i])


def detect_onset_offset(
    time_ms: np.ndarray, y: np.ndarray, env: np.ndarray, d: OnOffDef,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Single-trial onset + offset.
    Onset: forward or peak-anchored (selected by d.onset_method).
    Offset: peak-anchored forward — after env peak + gap, first sustained < thr.
    """
    if np.any(~np.isfinite(y)):
        return None, None

    dt = float(np.median(np.diff(time_ms)))
    mu, sd = _env_baseline_stats(env, time_ms)
    if sd < 1e-12 and mu < 1e-12:
        return None, None

    thr_on = mu + d.onset_k * max(sd, 1e-12)
    thr_off = mu + d.offset_k * max(sd, 1e-12)
    thr_backtrack = mu + d.onset_backtrack_k * max(sd, 1e-12)

    # --- Onset ---
    if d.onset_method == "forward":
        onset_ms = _detect_onset_forward(env, time_ms, thr_on, dt, d)
    elif d.onset_method == "peak_anchored":
        onset_ms = _detect_onset_peak_anchored(env, time_ms, thr_backtrack, dt, d)
    else:
        return None, None

    if onset_ms is None:
        return None, None

    # --- Offset (peak-anchored forward) ---
    seg_idx = np.flatnonzero(_mask(time_ms, (onset_ms, d.offset_search_end_ms)))
    if seg_idx.size == 0:
        return onset_ms, None

    peak_i = seg_idx[np.argmax(env[seg_idx])]
    start_off_ms = float(time_ms[peak_i]) + d.gap_ms

    oi = np.flatnonzero(_mask(time_ms, (start_off_ms, d.offset_search_end_ms)))
    if oi.size == 0:
        return onset_ms, None

    hold_n = max(1, round(d.hold_ms / dt))
    off_i = _first_run(oi, env[oi] < thr_off, hold_n)
    offset_ms = float(time_ms[off_i]) if off_i is not None else None

    return onset_ms, offset_ms


# ═══════════════════════════════════════════════════════════════════
#  Grid search — coverage-penalised scoring
# ═══════════════════════════════════════════════════════════════════
_SMOOTH_GRID = (0.5, 1.0, 2.0)
_ONSET_K_GRID = (2.5, 3.5, 5.0)
_OFFSET_K_GRID = (2.0, 3.0, 4.0)
_MIN_ABOVE_GRID = (0.75, 1.5)
_HOLD_GRID = (3.0, 5.0)
_GAP_GRID = (0.5, 1.5)
_MIN_ONSET_GRID = (10.0, 15.0, 20.0)
_BACKTRACK_K_GRID = (1.5, 2.5, 3.5)
_PEAK_SEARCH_GRID = ((15.0, 50.0), (15.0, 60.0))

_ONSET_SEARCH = (0.0, 60.0)
_OFFSET_SEARCH_END = 90.0


def _score_with_coverage(
    pred_on: np.ndarray, pred_off: np.ndarray,
    ds_on: np.ndarray, ds_off: np.ndarray,
    n_total: int, coverage_lambda: float,
) -> Tuple[float, int]:
    """score = MAE_onset + MAE_offset + λ × (1 - coverage)"""
    valid = np.isfinite(pred_on) & np.isfinite(pred_off)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return float("inf"), 0
    mae_on = _mae(pred_on[valid], ds_on[valid])
    mae_off = _mae(pred_off[valid], ds_off[valid])
    coverage = n_valid / n_total
    return mae_on + mae_off + coverage_lambda * (1.0 - coverage), n_valid


def _evaluate_onoff_params(
    time_ms: np.ndarray,
    envs_sub: np.ndarray,
    sigs_sub: np.ndarray,
    ds_on: np.ndarray,
    ds_off: np.ndarray,
    sig_name: str,
    smooth_ms: float,
    dt: float,
    coverage_lambda: float,
    min_valid: int = 30,
) -> Tuple[float, Optional[OnOffDef], int]:
    """
    For one (signal, smooth_ms): search both onset families × all param combos.
    Returns (best_score, best_def, n_valid).
    """
    n_trials = envs_sub.shape[0]
    best_score = float("inf")
    best_def: Optional[OnOffDef] = None
    best_nv = 0

    def _try(d: OnOffDef) -> None:
        nonlocal best_score, best_def, best_nv
        pred_on = np.full(n_trials, np.nan)
        pred_off = np.full(n_trials, np.nan)
        for j in range(n_trials):
            o, f = detect_onset_offset(time_ms, sigs_sub[j], envs_sub[j], d)
            if o is not None:
                pred_on[j] = o
            if f is not None:
                pred_off[j] = f

        score, nv = _score_with_coverage(
            pred_on, pred_off, ds_on, ds_off, n_trials, coverage_lambda,
        )
        if nv >= min_valid and score < best_score:
            best_score = score
            best_def = d
            best_nv = nv

    # --- Forward onset family ---
    for onset_k, offset_k, min_above, hold, gap, min_onset in itertools.product(
        _ONSET_K_GRID, _OFFSET_K_GRID, _MIN_ABOVE_GRID,
        _HOLD_GRID, _GAP_GRID, _MIN_ONSET_GRID,
    ):
        _try(OnOffDef(
            use_signal=sig_name, onset_method="forward",
            smooth_ms=smooth_ms,
            onset_k=onset_k, offset_k=offset_k,
            min_above_ms=min_above, onset_backtrack_k=onset_k,
            hold_ms=hold, gap_ms=gap,
            onset_search_ms=_ONSET_SEARCH,
            offset_search_end_ms=_OFFSET_SEARCH_END,
            min_onset_ms=min_onset,
            peak_search_ms=(15.0, 50.0),
        ))

    # --- Peak-anchored onset family ---
    for backtrack_k, offset_k, hold, gap, min_onset, peak_search in itertools.product(
        _BACKTRACK_K_GRID, _OFFSET_K_GRID,
        _HOLD_GRID, _GAP_GRID, _MIN_ONSET_GRID, _PEAK_SEARCH_GRID,
    ):
        _try(OnOffDef(
            use_signal=sig_name, onset_method="peak_anchored",
            smooth_ms=smooth_ms,
            onset_k=backtrack_k, offset_k=offset_k,
            min_above_ms=1.0, onset_backtrack_k=backtrack_k,
            hold_ms=hold, gap_ms=gap,
            onset_search_ms=_ONSET_SEARCH,
            offset_search_end_ms=_OFFSET_SEARCH_END,
            min_onset_ms=min_onset,
            peak_search_ms=peak_search,
        ))

    return best_score, best_def, best_nv


def infer_onset_offset_definition(
    df: pd.DataFrame, time_ms: np.ndarray,
    raw_bc: np.ndarray, filt_bc: np.ndarray,
    max_trials: int = 250,
    coverage_lambda: float = 30.0,
) -> Optional[OnOffDef]:
    """
    Grid-search with coverage-penalised scoring.
    With λ=30, missing 100% coverage costs 30ms of "error" — forcing the
    optimizer to explain most trials, not just an easy subset.
    """
    if "Onset" not in df.columns or "Offset" not in df.columns:
        return None

    ds_on = pd.to_numeric(df["Onset"], errors="coerce").to_numpy()
    ds_off = pd.to_numeric(df["Offset"], errors="coerce").to_numpy()
    ok = np.isfinite(ds_on) & np.isfinite(ds_off)

    if ok.sum() < 80:
        logger.warning("Not enough valid Onset/Offset (%d) to infer.", ok.sum())
        return None

    rng = np.random.default_rng(1)
    idx = np.flatnonzero(ok)
    if idx.size > max_trials:
        idx = rng.choice(idx, size=max_trials, replace=False)

    ds_on_sub = ds_on[idx]
    ds_off_sub = ds_off[idx]
    dt = float(np.median(np.diff(time_ms)))

    signals = {"raw_bc": raw_bc[idx], "filt_bc": filt_bc[idx]}
    best_score = float("inf")
    best_def: Optional[OnOffDef] = None
    best_nv = 0
    total_combos = len(signals) * len(_SMOOTH_GRID)
    progress = 0

    t0 = _time.monotonic()

    for sig_name, sig_sub in signals.items():
        for smooth_ms in _SMOOTH_GRID:
            progress += 1
            win_n = max(1, round(smooth_ms / dt))
            envs = rms_envelope_batch(sig_sub, win_n)

            score, d, nv = _evaluate_onoff_params(
                time_ms, envs, sig_sub, ds_on_sub, ds_off_sub,
                sig_name, smooth_ms, dt, coverage_lambda,
            )

            if d is not None and score < best_score:
                best_score = score
                best_def = d
                best_nv = nv

            elapsed = _time.monotonic() - t0
            logger.debug(
                "  search %d/%d  sig=%s smooth=%.2f  best=%.3f (nv=%d/%d)  %.1fs",
                progress, total_combos, sig_name, smooth_ms,
                best_score, best_nv, len(idx), elapsed,
            )

    elapsed = _time.monotonic() - t0

    if best_def is None:
        logger.warning("Could not infer Onset/Offset definition.")
        return None

    coverage = best_nv / len(idx)
    logger.info(
        "Onset/Offset inferred (%.1fs):\n"
        "  method=%s  sig=%s  smooth=%.2f ms\n"
        "  onset_k=%.1f  backtrack_k=%.1f  offset_k=%.1f\n"
        "  min_above=%.2f  hold=%.1f  gap=%.1f  min_onset=%.1f\n"
        "  peak_search=%s\n"
        "  score=%.3f  coverage=%.1f%% (%d/%d)",
        elapsed,
        best_def.onset_method, best_def.use_signal, best_def.smooth_ms,
        best_def.onset_k, best_def.onset_backtrack_k, best_def.offset_k,
        best_def.min_above_ms, best_def.hold_ms, best_def.gap_ms, best_def.min_onset_ms,
        best_def.peak_search_ms,
        best_score, 100 * coverage, best_nv, len(idx),
    )
    return best_def


# ═══════════════════════════════════════════════════════════════════
#  Label gate inference — when does the dataset assign Onset/Offset?
# ═══════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class LabelGate:
    """Rule for when to assign onset/offset (vs NaN)."""
    gate_type: str              # "threshold", "dual", or "none"
    gate_signal: str            # "ptp_raw", "ptp_filt", "env_zpeak_raw", "env_zpeak_filt", or ""
    threshold: float            # suppress when signal < this
    # Optional second gate (dual mode: label if EITHER fires)
    gate_signal_2: str = ""
    threshold_2: float = 0.0
    agreement: float = 0.0     # best agreement score


def _sweep_threshold(
    signal: np.ndarray, ds_has_label: np.ndarray, valid: np.ndarray,
) -> Tuple[float, float]:
    """Sweep thresholds on `signal`, return (best_threshold, best_agreement)."""
    candidates = np.percentile(signal[valid & np.isfinite(signal)], np.arange(1, 50))
    best_acc = 0.0
    best_thr = 0.0
    n_valid = float(valid.sum())
    for thr in candidates:
        pred = (signal >= thr) & valid
        acc = float(((pred == ds_has_label) & valid).sum()) / max(n_valid, 1)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def _sweep_threshold_weighted(
    signal: np.ndarray, ds_has_label: np.ndarray, valid: np.ndarray,
    w_fn: float = 3.0, w_fp: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Sweep thresholds minimising weighted FN + FP cost.
    Returns (best_threshold, best_cost, best_accuracy).
    FN = dataset labeled but we suppress (weight higher — we care about not losing labels).
    FP = dataset unlabeled but we label (weight lower).
    """
    fin = valid & np.isfinite(signal)
    if fin.sum() < 10:
        return 0.0, float("inf"), 0.0
    candidates = np.percentile(signal[fin], np.arange(1, 99))
    ds_label = ds_has_label.astype(np.int8)
    n_valid = float(valid.sum())
    best_cost = float("inf")
    best_thr = 0.0
    best_acc = 0.0
    for thr in candidates:
        pred = ((signal >= thr) & valid).astype(np.int8)
        fn = int(((ds_label == 1) & (pred == 0) & valid).sum())
        fp = int(((ds_label == 0) & (pred == 1) & valid).sum())
        cost = w_fn * fn + w_fp * fp
        acc = float(((pred == ds_label) & valid).sum()) / max(n_valid, 1)
        if cost < best_cost:
            best_cost = cost
            best_thr = float(thr)
            best_acc = acc
    return best_thr, best_cost, best_acc


def _compute_env_zpeak(
    signals: np.ndarray, time_ms: np.ndarray,
    smooth_ms: float, peak_search_ms: Tuple[float, float],
) -> np.ndarray:
    """Compute envelope peak z-score for a batch of signals."""
    dt = float(np.median(np.diff(time_ms)))
    win_n = max(1, round(smooth_ms / dt))
    envs = rms_envelope_batch(signals, win_n)

    bl_mask = _mask(time_ms, _BASELINE_WINDOW)
    peak_mask = _mask(time_ms, peak_search_ms)

    if not np.any(bl_mask) or not np.any(peak_mask):
        return np.full(signals.shape[0], np.nan)

    bl_envs = envs[:, bl_mask]
    bl_mu = np.mean(bl_envs, axis=1)
    bl_sd = np.std(bl_envs, axis=1, ddof=1)
    # Percentile-based noise floor: prevent z-score explosion when baseline
    # SD is near-zero (e.g. very quiet trials).  Use p20 of all trial SDs
    # as the floor so z-scores remain interpretable and comparable.
    finite_sd = bl_sd[np.isfinite(bl_sd)]
    floor = float(np.percentile(finite_sd, 20)) if finite_sd.size > 0 else 1e-12
    bl_sd = np.maximum(bl_sd, max(floor, 1e-12))
    peak_env = np.max(envs[:, peak_mask], axis=1)
    return (peak_env - bl_mu) / bl_sd


def _compute_env_auc(
    signals: np.ndarray, time_ms: np.ndarray,
    window_ms: Tuple[float, float],
) -> np.ndarray:
    """
    Area under rectified signal in a window, normalized by baseline RMS.
    Much less sensitive to single-sample spikes than peak z-score.
    """
    m = _mask(time_ms, window_ms)
    if not np.any(m):
        return np.full(signals.shape[0], np.nan)
    dt = float(np.median(np.diff(time_ms)))

    # Raw AUC (µV·ms)
    auc = np.sum(np.abs(signals[:, m]), axis=1) * dt

    # Normalize by baseline RMS to make it scale-invariant
    bl_mask = _mask(time_ms, _BASELINE_WINDOW)
    if not np.any(bl_mask):
        return auc  # return unnormalized if no baseline
    bl_rms = np.sqrt(np.mean(signals[:, bl_mask] ** 2, axis=1))
    finite_rms = bl_rms[np.isfinite(bl_rms)]
    floor = float(np.percentile(finite_rms, 20)) if finite_rms.size > 0 else 1e-12
    bl_rms = np.maximum(bl_rms, max(floor, 1e-12))
    return auc / bl_rms


def _compute_env_zpeak_wide(
    signals: np.ndarray, time_ms: np.ndarray,
    smooth_ms: float, peak_search_ms: Tuple[float, float],
    min_width_ms: float = 1.0,
) -> np.ndarray:
    """
    Width-constrained envelope peak z-score.
    Like env_zpeak but requires the envelope to stay above (mu + 2*sd) for
    at least min_width_ms around the peak.  Rejects single-sample spikes
    that inflate raw z-peak scores on unlabeled trials.
    """
    dt = float(np.median(np.diff(time_ms)))
    win_n = max(1, round(smooth_ms / dt))
    envs = rms_envelope_batch(signals, win_n)

    bl_mask = _mask(time_ms, _BASELINE_WINDOW)
    peak_mask = _mask(time_ms, peak_search_ms)

    if not np.any(bl_mask) or not np.any(peak_mask):
        return np.full(signals.shape[0], np.nan)

    bl_envs = envs[:, bl_mask]
    bl_mu = np.mean(bl_envs, axis=1)
    bl_sd = np.std(bl_envs, axis=1, ddof=1)
    finite_sd = bl_sd[np.isfinite(bl_sd)]
    floor = float(np.percentile(finite_sd, 20)) if finite_sd.size > 0 else 1e-12
    bl_sd = np.maximum(bl_sd, max(floor, 1e-12))

    peak_idx = np.flatnonzero(peak_mask)
    n_trials = signals.shape[0]
    result = np.full(n_trials, np.nan)
    min_width_n = max(1, round(min_width_ms / dt))

    for i in range(n_trials):
        env_i = envs[i, :]
        peak_local = np.argmax(env_i[peak_idx])
        peak_g = peak_idx[peak_local]
        peak_val = env_i[peak_g]

        # Width check: how many contiguous samples around peak are above
        # a moderate threshold (mu + 2*sd)
        width_thr = bl_mu[i] + 2.0 * bl_sd[i]
        if peak_val <= width_thr:
            continue  # peak not even above width threshold

        # Count contiguous above-threshold samples around peak
        count = 1
        # Forward
        j = peak_g + 1
        while j < len(env_i) and env_i[j] > width_thr:
            count += 1
            j += 1
        # Backward
        j = peak_g - 1
        while j >= 0 and env_i[j] > width_thr:
            count += 1
            j -= 1

        if count >= min_width_n:
            result[i] = (peak_val - bl_mu[i]) / bl_sd[i]

    return result


def _count_fn_fp(
    pred: np.ndarray, ds_label: np.ndarray, valid: np.ndarray,
) -> Tuple[int, int]:
    """Count false negatives and false positives."""
    p = pred.astype(np.int8)
    fn = int(((ds_label == 1) & (p == 0) & valid).sum())
    fp = int(((ds_label == 0) & (p == 1) & valid).sum())
    return fn, fp


def infer_label_gate(
    df: pd.DataFrame,
    time_ms: np.ndarray,
    raw_bc: np.ndarray,
    filt_bc: np.ndarray,
    ptp_def: Optional[PTPDef],
    onoff_def: Optional[OnOffDef],
    w_fn: float = 3.0,
    w_fp: float = 1.0,
    max_fp_increase: int = 5,
) -> LabelGate:
    """
    Infer the dataset's labeling policy by trying multiple gate signals:
      - PTP on raw_bc (in the inferred PTP window)
      - PTP on filt_bc (in the same window)
      - Envelope peak z-score on raw_bc
      - Envelope peak z-score on filt_bc

    Picks whichever signal + threshold gives lowest weighted cost
    (FN weighted higher than FP to avoid suppressing labeled trials).

    Also tries ALL-PAIRS dual gates (label if EITHER signal fires),
    with independent threshold sweeps per signal in each pair.
    """
    if "Onset" not in df.columns:
        return LabelGate(gate_type="none", gate_signal="", threshold=0.0)

    ds_has_onset = pd.to_numeric(df["Onset"], errors="coerce").notna().to_numpy()
    n = len(df)
    n_labeled = int(ds_has_onset.sum())
    n_unlabeled = n - n_labeled

    if n_labeled < 20 or n_unlabeled < 3:
        logger.info("Dataset labels onset on %d/%d trials — gate inference skipped.", n_labeled, n)
        return LabelGate(gate_type="none", gate_signal="", threshold=0.0)

    dt = float(np.median(np.diff(time_ms)))

    # Determine smoothing + peak search window from onoff_def (with fallbacks)
    smooth_ms = onoff_def.smooth_ms if onoff_def is not None else 2.0
    peak_search = (
        onoff_def.peak_search_ms
        if onoff_def is not None and onoff_def.onset_method == "peak_anchored"
        else (15.0, 50.0)
    )

    # Build candidate gate signals — always compute BOTH raw and filt variants
    gate_signals: Dict[str, np.ndarray] = {}

    # PTP signals
    if ptp_def is not None:
        m = _mask(time_ms, ptp_def.window_ms)
        if np.any(m):
            gate_signals["ptp_raw"] = np.ptp(raw_bc[:, m], axis=1)
            gate_signals["ptp_filt"] = np.ptp(filt_bc[:, m], axis=1)

    # Envelope peak z-score — BOTH raw and filt independently
    gate_signals["env_zpeak_raw"] = _compute_env_zpeak(
        raw_bc, time_ms, smooth_ms, peak_search,
    )
    gate_signals["env_zpeak_filt"] = _compute_env_zpeak(
        filt_bc, time_ms, smooth_ms, peak_search,
    )

    # Width-constrained z-peak on raw (rejects single-sample spikes)
    gate_signals["env_zpeak_raw_wide"] = _compute_env_zpeak_wide(
        raw_bc, time_ms, smooth_ms, peak_search, min_width_ms=3.0,
    )

    # AUC — area under rectified signal, normalized by baseline RMS
    # Less spike-sensitive than peak z-score; good "raw rescue" signal
    auc_window = (18.0, 45.0)
    gate_signals["auc_raw"] = _compute_env_auc(raw_bc, time_ms, auc_window)
    gate_signals["auc_filt"] = _compute_env_auc(filt_bc, time_ms, auc_window)

    # Drop any signal that's all-NaN
    gate_signals = {k: v for k, v in gate_signals.items() if np.any(np.isfinite(v))}

    if not gate_signals:
        return LabelGate(gate_type="none", gate_signal="", threshold=0.0)

    valid = np.ones(n, dtype=bool)  # all trials are valid candidates

    # Log distributions per signal
    for name, sig in gate_signals.items():
        labeled_vals = sig[ds_has_onset & np.isfinite(sig)]
        unlabeled_vals = sig[~ds_has_onset & np.isfinite(sig)]
        if labeled_vals.size > 0 and unlabeled_vals.size > 0:
            logger.info(
                "  gate signal '%s': labeled median=%.1f [p5=%.1f min=%.1f]  "
                "unlabeled median=%.1f [p95=%.1f max=%.1f]",
                name,
                float(np.median(labeled_vals)), float(np.percentile(labeled_vals, 5)),
                float(np.min(labeled_vals)),
                float(np.median(unlabeled_vals)), float(np.percentile(unlabeled_vals, 95)),
                float(np.max(unlabeled_vals)),
            )

    # Sweep each signal independently (weighted objective)
    n_valid_f = float(valid.sum())
    single_results: Dict[str, Tuple[float, float, float]] = {}  # name -> (thr, cost, acc)
    for name, sig in gate_signals.items():
        thr, cost, acc = _sweep_threshold_weighted(
            sig, ds_has_onset, valid, w_fn=w_fn, w_fp=w_fp,
        )
        single_results[name] = (thr, cost, acc)
        logger.info(
            "  gate '%s': threshold=%.2f  cost=%.1f  agreement=%.1f%%",
            name, thr, cost, 100 * acc,
        )

    # Pick best single gate (lowest cost)
    best_single_name = min(single_results, key=lambda k: single_results[k][1])
    best_thr, best_cost, best_acc = single_results[best_single_name]

    best_gate = LabelGate(
        gate_type="threshold", gate_signal=best_single_name,
        threshold=best_thr, agreement=best_acc,
    )

    logger.info(
        "  best single gate: '%s' >= %.2f  cost=%.1f  agreement=%.1f%%",
        best_single_name, best_thr, best_cost, 100 * best_acc,
    )

    # --- ALL-PAIRS dual gate search ---
    # Rules for accepting a dual gate:
    #   1. FN must be strictly lower than current best (dual is for rescuing misses)
    #   2. FP must not exceed current best FP + max_fp_increase
    #   3. Among qualifying candidates, pick lowest weighted cost
    # Both thresholds are swept (coarse grid) so the search isn't locked to
    # single-signal-optimal thresholds.
    sig_names = list(gate_signals.keys())
    ds_label = ds_has_onset.astype(np.int8)

    # Compute FN/FP for current best single gate
    _best_sig = gate_signals[best_single_name]
    _best_pred = (np.isfinite(_best_sig) & (_best_sig >= best_thr)).astype(np.int8)
    best_fn, best_fp = _count_fn_fp(_best_pred, ds_label, valid)

    logger.info(
        "  best single gate FN=%d FP=%d (max_fp_increase=%d)",
        best_fn, best_fp, max_fp_increase,
    )

    def _dual_thr_candidates(
        sig: np.ndarray, fin: np.ndarray,
    ) -> np.ndarray:
        """Build threshold candidates that reach the low end of the distribution.
        Includes overall percentiles (1,3,...,99), labeled-only low percentiles
        (1,3,...,49), and the absolute minimum — so rare outliers like trial 122
        are always reachable."""
        # Overall distribution with finer steps including very low end
        base = np.percentile(sig[fin], np.arange(1, 100, 2))  # 1,3,5,...,99

        # Labeled-only distribution: lets us reach below labeled p5
        fin_lab = fin & ds_has_onset & np.isfinite(sig)
        if fin_lab.sum() >= 10:
            lab = np.percentile(sig[fin_lab], np.arange(1, 51, 2))  # 1..49 on labeled
            base = np.concatenate([base, lab])

        # Always include absolute min so outliers aren't unsearchable
        base = np.concatenate([base, [np.min(sig[fin])]])

        return np.unique(base)

    for i_a, i_b in itertools.combinations(range(len(sig_names)), 2):
        nameA = sig_names[i_a]
        nameB = sig_names[i_b]
        sigA = gate_signals[nameA]
        sigB = gate_signals[nameB]
        finA = valid & np.isfinite(sigA)
        finB = valid & np.isfinite(sigB)
        if finA.sum() < 10 or finB.sum() < 10:
            continue

        thrA_candidates = _dual_thr_candidates(sigA, finA)
        thrB_candidates = _dual_thr_candidates(sigB, finB)

        for thrA in thrA_candidates:
            passesA = np.isfinite(sigA) & (sigA >= thrA)
            for thrB in thrB_candidates:
                pred_dual = (passesA | (np.isfinite(sigB) & (sigB >= thrB))).astype(np.int8)
                fn, fp = _count_fn_fp(pred_dual, ds_label, valid)

                # Must rescue at least one labeled-miss relative to current best
                if fn >= best_fn:
                    continue
                # FP cap relative to current best
                if fp > best_fp + max_fp_increase:
                    continue

                cost = w_fn * fn + w_fp * fp
                acc = float(((pred_dual == ds_label) & valid).sum()) / max(n_valid_f, 1.0)
                if cost < best_cost:
                    best_cost = cost
                    best_acc = acc
                    best_fn, best_fp = fn, fp
                    best_gate = LabelGate(
                        gate_type="dual",
                        gate_signal=nameA, threshold=float(thrA),
                        gate_signal_2=nameB, threshold_2=float(thrB),
                        agreement=best_acc,
                    )
                    logger.info(
                        "  dual improved (FN↓, FP capped): '%s' >= %.2f OR '%s' >= %.2f  "
                        "FN=%d FP=%d cost=%.1f agreement=%.1f%%",
                        nameA, float(thrA), nameB, float(thrB),
                        fn, fp, cost, 100 * acc,
                    )

    # Final summary
    if best_gate.gate_type == "dual":
        logger.info(
            "Label gate: DUAL  '%s' >= %.2f OR '%s' >= %.2f  (agreement=%.1f%%)",
            best_gate.gate_signal, best_gate.threshold,
            best_gate.gate_signal_2, best_gate.threshold_2,
            100 * best_gate.agreement,
        )
    else:
        logger.info(
            "Label gate: '%s' >= %.2f  (agreement=%.1f%%)",
            best_gate.gate_signal, best_gate.threshold, 100 * best_gate.agreement,
        )

    return best_gate


def _apply_label_gate(
    gate: LabelGate,
    our_on: np.ndarray,
    our_off: np.ndarray,
    gate_signals: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Suppress onset/offset on trials that don't pass the gate."""
    if gate.gate_type == "none":
        return our_on, our_off

    sig1 = gate_signals.get(gate.gate_signal)
    if sig1 is None:
        return our_on, our_off

    if gate.gate_type == "threshold":
        suppress = ~np.isfinite(sig1) | (sig1 < gate.threshold)
    elif gate.gate_type == "dual":
        sig2 = gate_signals.get(gate.gate_signal_2)
        passes_1 = np.isfinite(sig1) & (sig1 >= gate.threshold)
        passes_2 = (np.isfinite(sig2) & (sig2 >= gate.threshold_2)) if sig2 is not None else np.zeros_like(passes_1)
        suppress = ~(passes_1 | passes_2)
    else:
        return our_on, our_off

    our_on = our_on.copy()
    our_off = our_off.copy()
    our_on[suppress] = np.nan
    our_off[suppress] = np.nan
    return our_on, our_off


def _log_gate_mismatches(
    df: pd.DataFrame,
    our_on: np.ndarray,
    gate_signals: Dict[str, np.ndarray],
    max_show: int = 15,
) -> None:
    """Log trials where dataset has onset but our gate suppressed it (diagnostic)."""
    if "Onset" not in df.columns:
        return
    ds_has = pd.to_numeric(df["Onset"], errors="coerce").notna().to_numpy()
    we_suppressed = ~np.isfinite(our_on)
    mismatch = ds_has & we_suppressed
    n_mismatch = int(mismatch.sum())
    if n_mismatch == 0:
        logger.info("Gate mismatches: 0 (perfect agreement on labeled trials)")
        return

    logger.info("Gate mismatches: %d trials where dataset has onset but we suppressed:", n_mismatch)
    idx = np.flatnonzero(mismatch)[:max_show]
    for i in idx:
        parts = [f"  trial {i}"]
        for name, sig in gate_signals.items():
            parts.append(f"{name}={sig[i]:.1f}")
        if "PTP" in df.columns:
            ds_ptp = pd.to_numeric(df.loc[i, "PTP"], errors="coerce")
            if pd.notna(ds_ptp):
                parts.append(f"ds_PTP={ds_ptp:.1f}")
        ds_on = pd.to_numeric(df.loc[i, "Onset"], errors="coerce")
        parts.append(f"ds_Onset={ds_on:.1f}")
        logger.info("  ".join(parts))


# ═══════════════════════════════════════════════════════════════════
#  Compute final metrics
# ═══════════════════════════════════════════════════════════════════
def compute_metrics(
    df: pd.DataFrame, time_ms: np.ndarray,
    raw_bc: np.ndarray, filt_bc: np.ndarray,
    ptp_def: Optional[PTPDef], onoff_def: Optional[OnOffDef],
    label_gate: Optional[LabelGate] = None,
    response_ptp_uv: float = 50.0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    n = len(df)
    dt = float(np.median(np.diff(time_ms)))

    sig_ptp = raw_bc if (ptp_def is None or ptp_def.use_signal == "raw_bc") else filt_bc
    sig_onoff = raw_bc if (onoff_def is None or onoff_def.use_signal == "raw_bc") else filt_bc

    # PTP — vectorised
    our_ptp = np.full(n, np.nan)
    if ptp_def is not None:
        m = _mask(time_ms, ptp_def.window_ms)
        if np.any(m):
            our_ptp = np.ptp(sig_ptp[:, m], axis=1)

    # Onset / Offset
    our_on = np.full(n, np.nan)
    our_off = np.full(n, np.nan)
    if onoff_def is not None:
        win_n = max(1, round(onoff_def.smooth_ms / dt))
        envs = rms_envelope_batch(sig_onoff, win_n)
        for i in range(n):
            o, f = detect_onset_offset(time_ms, sig_onoff[i], envs[i], onoff_def)
            if o is not None:
                our_on[i] = o
            if f is not None:
                our_off[i] = f

    # Build gate signals dict for label gate
    gate_signals: Dict[str, np.ndarray] = {}
    if ptp_def is not None:
        m = _mask(time_ms, ptp_def.window_ms)
        if np.any(m):
            gate_signals["ptp_raw"] = np.ptp(raw_bc[:, m], axis=1)
            gate_signals["ptp_filt"] = np.ptp(filt_bc[:, m], axis=1)

    # Compute env_zpeak for both raw and filt
    smooth_ms = onoff_def.smooth_ms if onoff_def is not None else 2.0
    ps = (
        onoff_def.peak_search_ms
        if onoff_def is not None and onoff_def.onset_method == "peak_anchored"
        else (15.0, 50.0)
    )
    zpeak_raw = _compute_env_zpeak(raw_bc, time_ms, smooth_ms, ps)
    zpeak_filt = _compute_env_zpeak(filt_bc, time_ms, smooth_ms, ps)
    if np.any(np.isfinite(zpeak_raw)):
        gate_signals["env_zpeak_raw"] = zpeak_raw
    if np.any(np.isfinite(zpeak_filt)):
        gate_signals["env_zpeak_filt"] = zpeak_filt

    # Width-constrained z-peak on raw
    zpeak_raw_wide = _compute_env_zpeak_wide(raw_bc, time_ms, smooth_ms, ps, min_width_ms=3.0)
    if np.any(np.isfinite(zpeak_raw_wide)):
        gate_signals["env_zpeak_raw_wide"] = zpeak_raw_wide

    # AUC signals
    auc_window = (18.0, 45.0)
    auc_raw = _compute_env_auc(raw_bc, time_ms, auc_window)
    auc_filt = _compute_env_auc(filt_bc, time_ms, auc_window)
    if np.any(np.isfinite(auc_raw)):
        gate_signals["auc_raw"] = auc_raw
    if np.any(np.isfinite(auc_filt)):
        gate_signals["auc_filt"] = auc_filt

    # Apply label gate
    if label_gate is not None:
        our_on, our_off = _apply_label_gate(label_gate, our_on, our_off, gate_signals)
        _log_gate_mismatches(df, our_on, gate_signals)

    dur = our_off - our_on
    dur[~np.isfinite(dur) | (dur <= 0)] = np.nan

    result = df.copy()
    result["Our_PTP"] = our_ptp
    result["Our_Onset"] = our_on
    result["Our_Offset"] = our_off
    result["Our_Duration"] = dur
    result["Response_50uV"] = np.isfinite(our_ptp) & (our_ptp >= response_ptp_uv)

    for ds_col, our_col, err_col in [
        ("PTP", "Our_PTP", "PTP_Error"),
        ("Onset", "Our_Onset", "Onset_Error"),
        ("Offset", "Our_Offset", "Offset_Error"),
    ]:
        if ds_col in result.columns:
            result[err_col] = (
                pd.to_numeric(result[our_col], errors="coerce")
                - pd.to_numeric(result[ds_col], errors="coerce")
            )

    return result, sig_onoff


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════
def _savefig(fig, path: Path, dpi: int = 170) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.info("Saved %s", path.name)


def plot_error_hists(out: pd.DataFrame, save_dir: Path) -> None:
    cols = [c for c in ("PTP_Error", "Onset_Error", "Offset_Error") if c in out.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, c in zip(axes, cols):
        e = pd.to_numeric(out[c], errors="coerce").dropna()
        if e.empty:
            continue
        ax.hist(e, bins=40, edgecolor="white", linewidth=0.4)
        ax.axvline(0, ls="--", lw=0.8, color="black")
        med = float(np.median(e))
        ax.axvline(med, lw=1.0, color="red", label=f"median={med:.2f}")
        ax.set_title(c)
        ax.set_xlabel("computed − dataset")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    _savefig(fig, save_dir / "error_distributions_auto.png", dpi=160)


def plot_top_disagreements(
    out: pd.DataFrame, time_ms: np.ndarray, sig: np.ndarray,
    ptp_def: Optional[PTPDef], onoff_def: Optional[OnOffDef],
    save_dir: Path, n_show: int = 6,
) -> None:
    def _plot_metric(err_col: str, our_col: str, ds_col: str, label: str) -> None:
        if err_col not in out.columns:
            return
        errs = pd.to_numeric(out[err_col], errors="coerce").dropna()
        if errs.empty:
            return
        worst = errs.abs().nlargest(min(n_show, len(errs))).index.tolist()
        n = len(worst)
        fig, axes = plt.subplots(n, 1, figsize=(12, 2.6 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, i in zip(axes, worst):
            ax.plot(time_ms, sig[i], lw=0.8)
            ax.axvline(0, ls="--", lw=0.6, color="grey")

            if label == "PTP" and ptp_def is not None:
                ax.axvspan(*ptp_def.window_ms, alpha=0.10, color="orange", label="PTP window")

            our_v = pd.to_numeric(out.loc[i, our_col], errors="coerce")
            ds_v = pd.to_numeric(out.loc[i, ds_col], errors="coerce") if ds_col in out.columns else np.nan
            if pd.notna(our_v):
                ax.axvline(float(our_v), color="green", lw=1.4, label=f"Ours {our_v:.1f}")
            if pd.notna(ds_v):
                ax.axvline(float(ds_v), color="red", lw=1.4, ls="--", label=f"DS {ds_v:.1f}")

            ax.set_title(f"trial {i} | {err_col}={errs.loc[i]:+.2f}", fontsize=9)
            ax.set_ylabel("µV")
            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel("Time (ms)")
        _savefig(fig, save_dir / f"top_disagreements_{label.lower()}_auto.png")

    _plot_metric("PTP_Error", "Our_PTP", "PTP", "PTP")
    _plot_metric("Onset_Error", "Our_Onset", "Onset", "Onset")
    _plot_metric("Offset_Error", "Our_Offset", "Offset", "Offset")


def _log_error_summary(out: pd.DataFrame) -> None:
    for metric in ("PTP", "Onset", "Offset"):
        err_col = f"{metric}_Error"
        our_col = f"Our_{metric}"
        if err_col not in out.columns:
            continue
        e = pd.to_numeric(out[err_col], errors="coerce").dropna()
        if e.empty:
            continue

        ds_present = pd.to_numeric(out.get(metric, pd.Series(dtype=float)), errors="coerce").notna()
        detected = pd.to_numeric(out.get(our_col, pd.Series(dtype=float)), errors="coerce").notna()
        overlap = int((ds_present & detected).sum())
        n_ds = int(ds_present.sum())
        extra = int((~ds_present & detected).sum())

        logger.info(
            "%s: N=%d  MAE=%.3f  median=%.3f  [p25=%.3f  p75=%.3f]  "
            "overlap: %d/%d dataset (%.1f%%)  extra: %d",
            err_col, len(e), float(np.mean(np.abs(e))), float(np.median(e)),
            float(np.percentile(e, 25)), float(np.percentile(e, 75)),
            overlap, n_ds, 100 * overlap / max(n_ds, 1), extra,
        )


# ═══════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TMS-EMG auto reverse-engineer v2 (coverage-penalised, dual onset families)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("csv", type=Path, help="Input CSV path")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output CSV")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--max-fit-trials", type=int, default=250,
                    help="Max trials for grid search (speed vs accuracy)")
    p.add_argument("--coverage-lambda", type=float, default=30.0,
                    help="Coverage penalty weight (ms-scale). Higher = must explain more trials.")
    p.add_argument("--notch-hz", type=float, default=60.0, help="Mains frequency")
    p.add_argument("--no-filter", action="store_true", help="Skip bandpass + notch")
    p.add_argument("--gate-w-fn", type=float, default=3.0,
                    help="Weight for false negatives in gate search (suppressing a labeled trial)")
    p.add_argument("--gate-w-fp", type=float, default=1.0,
                    help="Weight for false positives in gate search (labeling an unlabeled trial)")
    p.add_argument("--gate-max-fp-increase", type=int, default=5,
                    help="Max extra FPs a dual gate may add over best single gate")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    csv_path = args.csv.expanduser().resolve()
    if not csv_path.exists():
        logger.error("File not found: %s", csv_path)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    logger.info("Loaded %s  (%d rows × %d cols)", csv_path.name, *df.shape)

    time_cols = find_time_columns(df)
    if len(time_cols) < 10:
        logger.error("Expected float-named time columns; found only %d.", len(time_cols))
        sys.exit(1)

    time_ms = time_axis_from_columns(time_cols)
    fs = infer_fs_hz(time_ms)
    dt = float(np.median(np.diff(time_ms)))
    logger.info("Time axis: %d cols  %.2f → %.2f ms  dt=%.4f ms  fs≈%.0f Hz",
                len(time_cols), time_ms[0], time_ms[-1], dt, fs)

    pcfg = PreprocConfig(do_filter=not args.no_filter, notch_hz=args.notch_hz)

    t0 = _time.monotonic()
    raw_bc, filt_bc = preprocess_all(df, time_cols, time_ms, fs, pcfg)
    logger.info("Preprocessed %d trials in %.2f s", len(df), _time.monotonic() - t0)

    # Auto-infer
    ptp_def = infer_ptp_definition(df, time_ms, raw_bc, filt_bc, max_trials=args.max_fit_trials)
    onoff_def = infer_onset_offset_definition(
        df, time_ms, raw_bc, filt_bc,
        max_trials=args.max_fit_trials,
        coverage_lambda=args.coverage_lambda,
    )

    # Infer label gate (needs PTP + envelope signals)
    label_gate = infer_label_gate(
        df, time_ms, raw_bc, filt_bc, ptp_def, onoff_def,
        w_fn=args.gate_w_fn, w_fp=args.gate_w_fp,
        max_fp_increase=args.gate_max_fp_increase,
    )

    # Compute
    out, sig_plots = compute_metrics(df, time_ms, raw_bc, filt_bc, ptp_def, onoff_def, label_gate)
    _log_error_summary(out)

    out_path = args.output or csv_path.with_name(csv_path.stem + "_analysed_auto.csv")
    out.to_csv(out_path, index=False)
    logger.info("Saved → %s", out_path)

    if not args.no_plots:
        save_dir = out_path.parent
        plot_error_hists(out, save_dir)
        plot_top_disagreements(out, time_ms, sig_plots, ptp_def, onoff_def, save_dir)


if __name__ == "__main__":
    main()