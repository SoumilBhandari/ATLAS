#!/usr/bin/env python3
"""
MEP classifier pipeline v5
-------------------------------------------------------------
Key changes from v4:
  1. TRUE per-subject normalization: for each subject, compute that subject's
     median/IQR of amplitude features across their trials. Normalize each trial
     by its own subject's stats. At test time, use the test subject's own stats
     (this is within-subject, so no leakage). Training subjects use their own.
  2. Dual operating-point reporting: every model is evaluated at BOTH
     "detection" (max_spec_at_recall, target=0.95) and "balanced" (max_bal_acc)
     thresholds in a single run. Comparison CSV shows both side-by-side.
  3. Simplified CLI — no more choosing threshold mode. Both are always computed.

Features (20):
  Original 13: ptp_raw/filt, zpeak_raw/filt, auc_raw/filt, peak_lat_raw/filt,
    baseline_rms_raw/filt, peak_to_base_raw/filt, width_ms_filt
  New 7: lat_dev_filt, n_peaks_filt, energy_ratio_filt, bp_lo/mid/hi_filt,
    bp_ratio_lo_hi_filt

Models: LogisticRegression, RandomForest, HistGradientBoosting
  All wrapped in CalibratedClassifierCV(isotonic, cv=3) by default.

Validation: LOSO with inner-fold threshold tuning (subject-level split)

Usage (from ATLAS/scripts/):
  python mep_classifier_pipeline.py -v
  python mep_classifier_pipeline.py --no-calibrate -v
  python mep_classifier_pipeline.py --models hgb -v
  python mep_classifier_pipeline.py --target-recall 0.97 -v
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, welch

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("mep_pipeline")


# ============================================================
# Config
# ============================================================
BASELINE_WIN = (-50.0, -10.0)
BLANK_WIN = (-1.0, 6.0)

BANDPASS = (20.0, 450.0)
BP_ORDER = 4
NOTCH_HZ = 60.0
NOTCH_Q = 30.0
NOTCH_HARMONICS = (2, 3)

ENV_SMOOTH_MS = 2.0

PTP_WIN = (18.0, 45.0)
AUC_WIN = (18.0, 45.0)
PEAK_WIN = (10.0, 60.0)
WIDTH_K = 3.0

EXPECTED_LATENCY_MS = 25.0

BP_LO = (20.0, 60.0)
BP_MID = (60.0, 150.0)
BP_HI = (150.0, 300.0)

INNER_VAL_FRAC = 0.2

# Amplitude features that get per-subject normalization
AMPLITUDE_FEATURES = [
    "ptp_raw", "ptp_filt", "auc_raw", "auc_filt",
    "baseline_rms_raw", "baseline_rms_filt",
    "energy_ratio_filt",
    "bp_lo_filt", "bp_mid_filt", "bp_hi_filt",
]


# ============================================================
# Utilities
# ============================================================
def find_time_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        try:
            float(c)
            cols.append(c)
        except (ValueError, TypeError):
            pass
    return sorted(cols, key=lambda c: float(c))


def time_axis_from_columns(time_cols: List[str]) -> np.ndarray:
    return np.array([float(c) for c in time_cols], dtype=np.float64)


def infer_fs_hz(time_ms: np.ndarray) -> float:
    if time_ms.size < 2:
        return 5000.0
    dt = float(np.median(np.diff(time_ms)))
    return 1000.0 / dt if dt > 0 else 5000.0


def mask_window(time_ms: np.ndarray, w: Tuple[float, float]) -> np.ndarray:
    return (time_ms >= w[0]) & (time_ms <= w[1])


def blank_artifact(time_ms: np.ndarray, y: np.ndarray, w: Tuple[float, float]) -> np.ndarray:
    out = y.copy()
    m = mask_window(time_ms, w)
    if not np.any(m):
        return out
    idx = np.flatnonzero(m)
    i0, i1 = int(idx[0]), int(idx[-1])
    left = max(i0 - 1, 0)
    right = min(i1 + 1, len(y) - 1)
    out[idx] = np.linspace(out[left], out[right], len(idx))
    return out


def baseline_correct(time_ms: np.ndarray, y: np.ndarray, w: Tuple[float, float]) -> np.ndarray:
    m = mask_window(time_ms, w)
    if not np.any(m):
        return y.copy()
    return y - float(np.mean(y[m]))


def build_filters(fs: float) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    nyq = fs / 2.0
    lo = max(0.1, BANDPASS[0]) / nyq
    hi = min(BANDPASS[1], nyq - 1.0) / nyq
    sos = butter(BP_ORDER, [lo, hi], btype="bandpass", output="sos")
    notch_filters: List[Tuple[np.ndarray, np.ndarray]] = []
    for h in (1, *NOTCH_HARMONICS):
        f0 = NOTCH_HZ * h
        if f0 < nyq:
            b, a = iirnotch(w0=f0, Q=NOTCH_Q, fs=fs)
            notch_filters.append((b, a))
    return sos, notch_filters


def apply_filters(y: np.ndarray, sos: np.ndarray,
                  notches: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    out = sosfiltfilt(sos, y)
    for b, a in notches:
        out = filtfilt(b, a, out)
    return out


def rms_envelope(y: np.ndarray, win_n: int) -> np.ndarray:
    win_n = max(1, int(win_n))
    n = y.size
    cs2 = np.empty(n + 1, dtype=np.float64)
    cs2[0] = 0.0
    np.cumsum(y * y, out=cs2[1:])
    idx = np.arange(n)
    lo = np.maximum(idx + 1 - win_n, 0)
    counts = (idx + 1 - lo).astype(np.float64)
    return np.sqrt((cs2[idx + 1] - cs2[lo]) / counts)


def longest_run_true(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def count_peaks_above(env: np.ndarray, threshold: float) -> int:
    above = env > threshold
    if not np.any(above):
        return 0
    count = 0
    in_peak = False
    for v in above:
        if v and not in_peak:
            count += 1
            in_peak = True
        elif not v:
            in_peak = False
    return count


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    weight_map = {c: n / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    return np.array([weight_map[yi] for yi in y], dtype=np.float64)


def bandpower(signal: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    nperseg = min(len(signal), max(16, int(fs * 0.01)))
    if nperseg < 4:
        return np.nan
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return np.nan
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    return float(_trapz(psd[mask], freqs[mask]))


# ============================================================
# Per-trial feature extraction (20 features)
# ============================================================
def compute_trial_features(
    time_ms: np.ndarray,
    raw_bc: np.ndarray,
    filt_bc: np.ndarray,
    env_raw: np.ndarray,
    env_filt: np.ndarray,
    fs: float,
) -> Dict[str, float]:
    dt = float(np.median(np.diff(time_ms)))

    m_ptp = mask_window(time_ms, PTP_WIN)
    m_auc = mask_window(time_ms, AUC_WIN)
    m_peak = mask_window(time_ms, PEAK_WIN)
    m_bl = mask_window(time_ms, BASELINE_WIN)

    feats: Dict[str, float] = {}

    # PTP
    feats["ptp_raw"] = float(np.ptp(raw_bc[m_ptp])) if np.any(m_ptp) else np.nan
    feats["ptp_filt"] = float(np.ptp(filt_bc[m_ptp])) if np.any(m_ptp) else np.nan

    # Baseline RMS
    feats["baseline_rms_raw"] = float(np.sqrt(np.mean(raw_bc[m_bl] ** 2))) if np.any(m_bl) else np.nan
    feats["baseline_rms_filt"] = float(np.sqrt(np.mean(filt_bc[m_bl] ** 2))) if np.any(m_bl) else np.nan

    # Envelope baseline stats
    bl_env_raw = env_raw[m_bl] if np.any(m_bl) else np.array([np.nan])
    bl_env_filt = env_filt[m_bl] if np.any(m_bl) else np.array([np.nan])
    mu_raw = float(np.nanmean(bl_env_raw))
    sd_raw = max(float(np.nanstd(bl_env_raw, ddof=1)),
                 0.05 * abs(mu_raw) if np.isfinite(mu_raw) else 0.0, 1e-3)
    mu_filt = float(np.nanmean(bl_env_filt))
    sd_filt = max(float(np.nanstd(bl_env_filt, ddof=1)),
                  0.05 * abs(mu_filt) if np.isfinite(mu_filt) else 0.0, 1e-3)

    # Peak envelope + latency
    if np.any(m_peak):
        peak_idx = np.flatnonzero(m_peak)
        i_raw = int(peak_idx[np.argmax(env_raw[m_peak])])
        i_filt = int(peak_idx[np.argmax(env_filt[m_peak])])
        peak_env_raw = float(env_raw[i_raw])
        peak_env_filt = float(env_filt[i_filt])
        feats["peak_lat_raw"] = float(time_ms[i_raw])
        feats["peak_lat_filt"] = float(time_ms[i_filt])
    else:
        peak_env_raw = peak_env_filt = np.nan
        feats["peak_lat_raw"] = feats["peak_lat_filt"] = np.nan

    # zpeak
    feats["zpeak_raw"] = float((peak_env_raw - mu_raw) / sd_raw) if np.isfinite(peak_env_raw) else np.nan
    feats["zpeak_filt"] = float((peak_env_filt - mu_filt) / sd_filt) if np.isfinite(peak_env_filt) else np.nan

    # peak_to_base
    feats["peak_to_base_raw"] = float(peak_env_raw / mu_raw) if (np.isfinite(peak_env_raw) and abs(mu_raw) > 1e-9) else np.nan
    feats["peak_to_base_filt"] = float(peak_env_filt / mu_filt) if (np.isfinite(peak_env_filt) and abs(mu_filt) > 1e-9) else np.nan

    # AUC rectified, normalized by baseline RMS
    if np.any(m_auc):
        auc_raw_val = float(np.sum(np.abs(raw_bc[m_auc])) * dt)
        auc_filt_val = float(np.sum(np.abs(filt_bc[m_auc])) * dt)
        d_raw = max(float(feats["baseline_rms_raw"]), 1e-6) if np.isfinite(feats["baseline_rms_raw"]) else 1e-6
        d_filt = max(float(feats["baseline_rms_filt"]), 1e-6) if np.isfinite(feats["baseline_rms_filt"]) else 1e-6
        feats["auc_raw"] = auc_raw_val / d_raw
        feats["auc_filt"] = auc_filt_val / d_filt
    else:
        feats["auc_raw"] = feats["auc_filt"] = np.nan

    # Width above threshold (filtered)
    if np.any(m_peak):
        thr = mu_filt + WIDTH_K * sd_filt
        run_len = longest_run_true(env_filt[m_peak] > thr)
        feats["width_ms_filt"] = float(run_len * dt)
    else:
        feats["width_ms_filt"] = np.nan

    # Latency deviation
    feats["lat_dev_filt"] = abs(feats["peak_lat_filt"] - EXPECTED_LATENCY_MS) if np.isfinite(feats.get("peak_lat_filt", np.nan)) else np.nan

    # Number of peaks above threshold
    if np.any(m_peak):
        feats["n_peaks_filt"] = float(count_peaks_above(env_filt[m_peak], mu_filt + WIDTH_K * sd_filt))
    else:
        feats["n_peaks_filt"] = np.nan

    # Energy ratio
    if np.any(m_auc) and np.any(m_bl):
        resp_e = float(np.sum(filt_bc[m_auc] ** 2) * dt)
        bl_e = float(np.sum(filt_bc[m_bl] ** 2) * dt)
        feats["energy_ratio_filt"] = resp_e / max(bl_e, 1e-9)
    else:
        feats["energy_ratio_filt"] = np.nan

    # Spectral bandpower
    if np.any(m_peak):
        sig_peak = filt_bc[m_peak]
        feats["bp_lo_filt"] = bandpower(sig_peak, fs, BP_LO)
        feats["bp_mid_filt"] = bandpower(sig_peak, fs, BP_MID)
        feats["bp_hi_filt"] = bandpower(sig_peak, fs, BP_HI)
    else:
        feats["bp_lo_filt"] = feats["bp_mid_filt"] = feats["bp_hi_filt"] = np.nan

    # Bandpower ratio
    if np.isfinite(feats.get("bp_lo_filt", np.nan)) and np.isfinite(feats.get("bp_hi_filt", np.nan)):
        feats["bp_ratio_lo_hi_filt"] = feats["bp_lo_filt"] / max(feats["bp_hi_filt"], 1e-12)
    else:
        feats["bp_ratio_lo_hi_filt"] = np.nan

    return feats


# ============================================================
# Loading + preprocessing
# ============================================================
def load_subject_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def preprocess_subject(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_cols = find_time_columns(df)
    if len(time_cols) < 10:
        raise ValueError("Not enough float-named time columns found.")
    time_ms = time_axis_from_columns(time_cols)
    fs = infer_fs_hz(time_ms)
    dt = float(np.median(np.diff(time_ms)))
    waves = df[time_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    n = waves.shape[0]
    sos, notches = build_filters(fs)
    raw_bc = np.empty_like(waves)
    filt_bc = np.empty_like(waves)
    for i in range(n):
        y = waves[i]
        if np.any(~np.isfinite(y)):
            y = pd.Series(y).interpolate(limit_direction="both").to_numpy(dtype=np.float64)
        yb = blank_artifact(time_ms, y, BLANK_WIN)
        raw_bc[i] = baseline_correct(time_ms, yb, BASELINE_WIN)
        yf = apply_filters(yb, sos, notches)
        filt_bc[i] = baseline_correct(time_ms, yf, BASELINE_WIN)
    win_n = max(1, int(round(ENV_SMOOTH_MS / dt)))
    env_raw = np.vstack([rms_envelope(raw_bc[i], win_n) for i in range(n)])
    env_filt = np.vstack([rms_envelope(filt_bc[i], win_n) for i in range(n)])
    return df, time_ms, fs, raw_bc, filt_bc, env_raw, env_filt


def label_from_onset_offset(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    onset = pd.to_numeric(df.get("Onset", np.nan), errors="coerce").to_numpy()
    offset = pd.to_numeric(df.get("Offset", np.nan), errors="coerce").to_numpy()
    has_on = np.isfinite(onset)
    has_off = np.isfinite(offset)
    dur = offset - onset
    y = (has_on & has_off & (dur > 0)).astype(np.int8)
    edge = {
        "onset_only": int((has_on & ~has_off).sum()),
        "offset_only": int((~has_on & has_off).sum()),
        "neg_or_zero_dur": int((has_on & has_off & (dur <= 0)).sum()),
        "both_nan": int((~has_on & ~has_off).sum()),
    }
    return y, edge


# ============================================================
# True per-subject feature normalization
# ============================================================
def per_subject_normalize(
    feats_df: pd.DataFrame,
    feature_cols: List[str],
    amp_cols: List[str],
) -> pd.DataFrame:
    """
    For each subject, normalize amplitude features by that subject's own
    median and IQR (across that subject's trials). This removes cross-subject
    amplitude variability due to physiology / electrode placement.

    No leakage: each subject's stats come only from their own trials.
    """
    df = feats_df.copy()
    for col in amp_cols:
        if col not in feature_cols or col not in df.columns:
            continue
        # Per-subject median and IQR
        grouped = df.groupby("subject_id")[col]
        med = grouped.transform("median")
        q75 = grouped.transform(lambda x: x.quantile(0.75))
        q25 = grouped.transform(lambda x: x.quantile(0.25))
        iqr = (q75 - q25).clip(lower=1e-6)
        df[col] = (df[col] - med) / iqr
    return df


# ============================================================
# Threshold tuning
# ============================================================
def _tune_on_probs(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    mode: str,
    target_recall: float = 0.95,
    n_grid: int = 201,
) -> float:
    thresholds = np.linspace(0.0, 1.0, n_grid)

    if mode == "max_bal_acc":
        best_thr, best_score = 0.5, -1.0
        for thr in thresholds:
            pred = (prob_val >= thr).astype(np.int8)
            score = float(balanced_accuracy_score(y_val, pred))
            if score > best_score:
                best_score = score
                best_thr = float(thr)
        return best_thr

    elif mode == "max_spec_at_recall":
        if (y_val == 1).sum() == 0 or (y_val == 0).sum() == 0:
            return 0.5
        best_thr, best_spec = 0.0, -1.0
        for thr in thresholds:
            pred = (prob_val >= thr).astype(np.int8)
            tp = int(np.sum((y_val == 1) & (pred == 1)))
            fn = int(np.sum((y_val == 1) & (pred == 0)))
            tn = int(np.sum((y_val == 0) & (pred == 0)))
            fp = int(np.sum((y_val == 0) & (pred == 1)))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            if recall >= target_recall and spec > best_spec:
                best_spec = spec
                best_thr = float(thr)
        return best_thr if best_spec >= 0 else 0.3

    else:  # fixed_0.5
        return 0.5


def tune_threshold_inner_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    subjects_train: np.ndarray,
    model,
    mode: str,
    target_recall: float,
    model_name: str,
) -> float:
    if mode == "fixed_0.5":
        return 0.5

    unique_subs = np.unique(subjects_train)

    if len(unique_subs) < 3:
        model.fit(X_train, y_train)
        prob = _get_probs(model, X_train)
        return _tune_on_probs(y_train, prob, mode, target_recall)

    n_val = max(1, int(round(len(unique_subs) * INNER_VAL_FRAC)))
    rng = np.random.RandomState(42)
    shuffled = rng.permutation(unique_subs)
    val_subs = set(shuffled[:n_val])

    inner_val_mask = np.isin(subjects_train, list(val_subs))
    inner_train_mask = ~inner_val_mask

    X_it, y_it = X_train[inner_train_mask], y_train[inner_train_mask]
    X_iv, y_iv = X_train[inner_val_mask], y_train[inner_val_mask]

    if len(np.unique(y_it)) < 2 or len(np.unique(y_iv)) < 2:
        model.fit(X_train, y_train)
        prob = _get_probs(model, X_train)
        return _tune_on_probs(y_train, prob, mode, target_recall)

    _fit_model(model, X_it, y_it, model_name)
    prob_iv = _get_probs(model, X_iv)
    return _tune_on_probs(y_iv, prob_iv, mode, target_recall)


def _get_probs(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    d = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


def _fit_model(model, X: np.ndarray, y: np.ndarray, model_name: str):
    # CalibratedClassifierCV doesn't support nested sample_weight,
    # so we just fit normally. Class weighting is handled inside each
    # base estimator (logreg: class_weight="balanced", rf: balanced_subsample).
    model.fit(X, y)


# ============================================================
# Metrics
# ============================================================
def compute_fold_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    out: Dict[str, float] = {}
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    out["recall"] = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    out["fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else np.nan
    out["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else np.nan
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    out["tn"] = float(tn)
    out["fp"] = float(fp)
    out["fn"] = float(fn)
    out["tp"] = float(tp)
    return out


def balanced_eval(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray,
                  rng: np.random.RandomState) -> Optional[Dict[str, float]]:
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return None
    n_min = min(len(pos_idx), len(neg_idx))
    if len(pos_idx) > n_min:
        pos_idx = rng.choice(pos_idx, size=n_min, replace=False)
    elif len(neg_idx) > n_min:
        neg_idx = rng.choice(neg_idx, size=n_min, replace=False)
    idx = np.sort(np.concatenate([pos_idx, neg_idx]))
    return compute_fold_metrics(y_true[idx], y_prob[idx], y_pred[idx])


# ============================================================
# Models
# ============================================================
def make_models(random_state: int = 0, calibrate: bool = True) -> Dict[str, object]:
    models = {}

    base_logreg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
    ])
    base_rf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=400, random_state=random_state,
                                       class_weight="balanced_subsample", n_jobs=-1)),
    ])
    base_hgb = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, max_iter=300,
                                               min_samples_leaf=20, random_state=random_state)),
    ])

    if calibrate:
        models["logreg"] = CalibratedClassifierCV(base_logreg, method="isotonic", cv=3)
        models["rf"] = CalibratedClassifierCV(base_rf, method="isotonic", cv=3)
        models["hgb"] = CalibratedClassifierCV(base_hgb, method="isotonic", cv=3)
    else:
        models["logreg"] = base_logreg
        models["rf"] = base_rf
        models["hgb"] = base_hgb

    return models


# ============================================================
# LOSO with dual thresholds
# ============================================================
def run_loso_dual(
    feats_df: pd.DataFrame,
    feature_cols: List[str],
    model,
    model_name: str,
    target_recall: float = 0.95,
    calibrated: bool = True,
) -> Dict[str, object]:
    """
    Run LOSO with TWO thresholds per fold:
      - "detection": max_spec_at_recall (high recall, best specificity possible)
      - "balanced": max_bal_acc (equal weight to sensitivity/specificity)

    Returns dict with results for both operating points.
    """
    X = feats_df[feature_cols].to_numpy(dtype=np.float64)
    y_all = feats_df["y"].to_numpy(dtype=np.int8)
    subjects = feats_df["subject_id"].to_numpy(dtype=np.int32)
    unique_sids = sorted(np.unique(subjects))

    # Results containers for each operating point
    modes = ["detection", "balanced"]
    results = {}
    for mode in modes:
        results[mode] = {
            "metrics_rows": [], "bal_metrics_rows": [],
            "y_true": [], "y_prob": [], "y_pred": [],
            "cm": np.zeros((2, 2), dtype=np.int64),
            "thresholds": [],
        }

    bal_rng = np.random.RandomState(99)

    for sid in unique_sids:
        test_mask = subjects == sid
        train_mask = ~test_mask
        X_train, y_train = X[train_mask], y_all[train_mask]
        X_test, y_test = X[test_mask], y_all[test_mask]
        subs_train = subjects[train_mask]

        # Tune BOTH thresholds on inner split (same inner split for consistency)
        thr_detect = tune_threshold_inner_split(
            X_train, y_train, subs_train, model,
            mode="max_spec_at_recall", target_recall=target_recall,
            model_name=model_name,
        )
        thr_balanced = tune_threshold_inner_split(
            X_train, y_train, subs_train, model,
            mode="max_bal_acc", target_recall=target_recall,
            model_name=model_name,
        )

        # Refit on full training set
        _fit_model(model, X_train, y_train, model_name)
        prob_test = _get_probs(model, X_test)

        # Evaluate at both thresholds
        for mode_name, thr in [("detection", thr_detect), ("balanced", thr_balanced)]:
            r = results[mode_name]
            pred = (prob_test >= thr).astype(np.int8)

            fold_m = compute_fold_metrics(y_test, prob_test, pred)
            fold_m["subject_id"] = int(sid)
            fold_m["n_test"] = int(test_mask.sum())
            fold_m["threshold"] = thr
            fold_m["model"] = model_name
            fold_m["mode"] = mode_name
            r["metrics_rows"].append(fold_m)

            bal_m = balanced_eval(y_test, prob_test, pred, bal_rng)
            if bal_m is not None:
                bal_m["subject_id"] = int(sid)
                r["bal_metrics_rows"].append(bal_m)

            cm = confusion_matrix(y_test, pred, labels=[0, 1])
            r["cm"] += cm
            r["thresholds"].append(thr)
            r["y_true"].append(y_test)
            r["y_prob"].append(prob_test)
            r["y_pred"].append(pred)

        logger.info(
            "LOSO [%s] subject %d | n=%d | detect: thr=%.3f rec=%.3f spec=%.3f | "
            "balanced: thr=%.3f rec=%.3f spec=%.3f",
            model_name, sid, int(test_mask.sum()),
            thr_detect,
            results["detection"]["metrics_rows"][-1].get("recall", np.nan),
            results["detection"]["metrics_rows"][-1].get("specificity", np.nan),
            thr_balanced,
            results["balanced"]["metrics_rows"][-1].get("recall", np.nan),
            results["balanced"]["metrics_rows"][-1].get("specificity", np.nan),
        )

    # Finalize
    for mode in modes:
        r = results[mode]
        r["metrics_df"] = pd.DataFrame(r["metrics_rows"]).sort_values("subject_id")
        r["bal_metrics_df"] = pd.DataFrame(r["bal_metrics_rows"]).sort_values("subject_id") if r["bal_metrics_rows"] else pd.DataFrame()
        r["y_true"] = np.concatenate(r["y_true"])
        r["y_prob"] = np.concatenate(r["y_prob"])
        r["y_pred"] = np.concatenate(r["y_pred"])
        r["thresholds"] = np.array(r["thresholds"])

    return results


# ============================================================
# Plotting
# ============================================================
def plot_confusion(cm: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Absent", "Present"]); ax.set_yticklabels(["Absent", "Present"])
    for (i, j), v in np.ndenumerate(cm):
        color = "white" if v > cm.max() / 2 else "black"
        ax.text(j, i, f"{int(v)}", ha="center", va="center", color=color, fontsize=13)
    fig.tight_layout(); fig.savefig(out_path, dpi=170); plt.close(fig)


def plot_pr_roc(y_true: np.ndarray, y_prob: np.ndarray, title: str, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax1.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc_val:.3f}")
        ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax1.set_title(f"ROC — {title}"); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
        ax1.legend(loc="lower right"); ax1.grid(alpha=0.3)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax2.plot(rec, prec, linewidth=2, label=f"AP = {ap:.3f}")
        ax2.axhline(y_true.mean(), color="gray", linestyle="--", linewidth=0.8, label=f"Prev={y_true.mean():.2f}")
        ax2.set_title(f"PR — {title}"); ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
        ax2.legend(loc="lower left"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=170); plt.close(fig)


def plot_threshold_hist(thrs: np.ndarray, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(thrs, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(thrs.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean={thrs.mean():.3f}")
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, label="0.5")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Count"); ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=170); plt.close(fig)


def plot_dual_comparison(all_comparison: List[Dict], out_path: Path) -> None:
    """Bar chart: detection vs balanced for each model, key metrics."""
    df = pd.DataFrame(all_comparison)
    metrics_to_plot = ["mean_recall", "mean_specificity", "mean_balanced_accuracy"]
    models = df["model"].unique()
    modes = df["mode"].unique()

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    colors = {"detection": "#2196F3", "balanced": "#FF9800"}

    for ax, metric in zip(axes, metrics_to_plot):
        x = np.arange(len(models))
        width = 0.35
        for i, mode in enumerate(modes):
            subset = df[df["mode"] == mode]
            vals = [float(subset[subset["model"] == m][metric].iloc[0]) if len(subset[subset["model"] == m]) > 0 else 0 for m in models]
            bars = ax.bar(x + i * width - width / 2, vals, width, label=mode, color=colors.get(mode, None))
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(models)
        ax.set_ylabel(metric.replace("mean_", "")); ax.set_ylim(0, 1.08)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        ax.set_title(metric.replace("mean_", ""))

    fig.suptitle("Detection vs Balanced — All Models", fontsize=12)
    fig.tight_layout(); fig.savefig(out_path, dpi=170); plt.close(fig)


def plot_subject_diagnostics(all_metrics: Dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, mdf in all_metrics.items():
        sdf = mdf.sort_values("subject_id")
        sids = sdf["subject_id"].values
        if "fnr" in sdf.columns:
            axes[0].plot(sids, sdf["fnr"].values, "o-", markersize=3, label=label)
        if "recall" in sdf.columns:
            axes[1].plot(sids, sdf["recall"].values, "o-", markersize=3, label=label)
    axes[0].set_title("FNR by Subject"); axes[0].set_xlabel("Subject ID"); axes[0].set_ylabel("FNR")
    axes[0].axhline(0.1, color="red", linestyle="--", linewidth=0.8, label="FNR=0.10")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_title("Recall by Subject"); axes[1].set_xlabel("Subject ID"); axes[1].set_ylabel("Recall")
    axes[1].axhline(0.9, color="red", linestyle="--", linewidth=0.8, label="Recall=0.90")
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=170); plt.close(fig)


# ============================================================
# Summary helper
# ============================================================
def summarize_mode(model_name: str, mode: str, r: Dict, target_recall: float) -> Dict:
    mdf = r["metrics_df"]
    cm = r["cm"]
    thrs = r["thresholds"]
    tn, fp, fn, tp = cm.ravel()

    agg_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    agg_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    agg_spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    agg_f1 = 2 * agg_prec * agg_rec / (agg_prec + agg_rec) if (agg_prec + agg_rec) > 0 else 0

    y_t, y_p = r["y_true"], r["y_prob"]
    if len(np.unique(y_t)) == 2:
        pooled_roc = float(roc_auc_score(y_t, y_p))
        pooled_pr = float(average_precision_score(y_t, y_p))
    else:
        pooled_roc = pooled_pr = np.nan

    bal_df = r.get("bal_metrics_df", pd.DataFrame())
    bal_spec = float(bal_df["specificity"].mean()) if not bal_df.empty and "specificity" in bal_df.columns else np.nan

    return {
        "model": model_name, "mode": mode,
        "target_recall": target_recall if mode == "detection" else np.nan,
        "mean_threshold": float(thrs.mean()), "std_threshold": float(thrs.std()),
        "agg_tn": int(tn), "agg_fp": int(fp), "agg_fn": int(fn), "agg_tp": int(tp),
        "agg_precision": agg_prec, "agg_recall": agg_rec, "agg_specificity": agg_spec, "agg_f1": agg_f1,
        "pooled_roc_auc": pooled_roc, "pooled_pr_auc": pooled_pr,
        "mean_balanced_accuracy": float(mdf["balanced_accuracy"].mean()),
        "mean_f1": float(mdf["f1"].mean()),
        "mean_recall": float(mdf["recall"].mean()),
        "mean_specificity": float(mdf["specificity"].mean()),
        "mean_fnr": float(mdf["fnr"].mean()),
        "mean_fpr": float(mdf["fpr"].mean()),
        "mean_pr_auc": float(mdf["pr_auc"].mean()),
        "mean_roc_auc": float(mdf["roc_auc"].mean()),
        "balanced_eval_spec": bal_spec,
    }


# ============================================================
# Main
# ============================================================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MEP classifier v5 — dual operating points")
    p.add_argument("--data-dir", type=Path, default=Path("../data/MEP_Data"))
    p.add_argument("--out-dir", type=Path, default=Path("../outputs"))
    p.add_argument("--models", nargs="+", default=["logreg", "rf", "hgb"],
                   choices=["logreg", "rf", "hgb"])
    p.add_argument("--target-recall", type=float, default=0.95,
                   help="Target recall for detection mode (default: 0.95)")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    do_calibrate = not args.no_calibrate
    do_normalize = not args.no_normalize

    data_dir = args.data_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover + load + preprocess ─────────────────────────
    paths = sorted(data_dir.glob("*.csv"), key=lambda p: int(p.stem))
    if not paths:
        raise SystemExit(f"No subject CSVs found in {data_dir}")
    logger.info("Found %d subject files in %s", len(paths), data_dir)

    rows: List[Dict] = []
    edge_rows: List[Dict] = []

    for path in paths:
        sid = int(path.stem)
        logger.info("Loading subject %d …", sid)
        df = load_subject_csv(path)
        y, edge = label_from_onset_offset(df)
        edge_rows.append({"subject_id": sid, **edge, "n_trials": len(df),
                          "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())})
        df0, time_ms, fs, raw_bc, filt_bc, env_raw, env_filt = preprocess_subject(df)
        for i in range(len(df0)):
            feats = compute_trial_features(time_ms, raw_bc[i], filt_bc[i], env_raw[i], env_filt[i], fs)
            rows.append({"subject_id": sid, "trial_idx": i, "y": int(y[i]), **feats})

    feats_df = pd.DataFrame(rows)
    edges_df = pd.DataFrame(edge_rows)

    n_pos = int((feats_df["y"] == 1).sum())
    n_neg = int((feats_df["y"] == 0).sum())
    logger.info("Overall: %d trials | pos=%d (%.1f%%) neg=%d (%.1f%%)",
                len(feats_df), n_pos, 100 * n_pos / len(feats_df), n_neg, 100 * n_neg / len(feats_df))

    edges_df["pos_frac"] = edges_df["n_pos"] / edges_df["n_trials"].clip(lower=1)
    extreme = edges_df[(edges_df["pos_frac"] > 0.95) | (edges_df["pos_frac"] < 0.05)]
    if len(extreme) > 0:
        logger.warning("Extreme imbalance subjects:\n%s",
                       extreme[["subject_id", "n_trials", "n_pos", "pos_frac"]].to_string(index=False))

    feats_df.to_csv(out_dir / "features_raw.csv", index=False)
    edges_df.to_csv(out_dir / "label_edge_cases_by_subject.csv", index=False)

    # ── Per-subject normalization ────────────────────────────
    feature_cols = [c for c in feats_df.columns if c not in ("subject_id", "trial_idx", "y")]
    amp_cols = [c for c in AMPLITUDE_FEATURES if c in feature_cols]

    if do_normalize and len(amp_cols) > 0:
        logger.info("Applying per-subject normalization to %d amplitude features", len(amp_cols))
        feats_df = per_subject_normalize(feats_df, feature_cols, amp_cols)
    else:
        logger.info("Skipping per-subject normalization")

    feats_df.to_csv(out_dir / "features.csv", index=False)
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols)
    logger.info("Calibrate: %s | Normalize: %s | Target recall: %.2f",
                do_calibrate, do_normalize, args.target_recall)

    # ── Run LOSO for each model (dual thresholds) ────────────
    all_models = make_models(calibrate=do_calibrate)
    all_comparison = []
    all_detect_metrics = {}
    all_balanced_metrics = {}

    for model_name in args.models:
        logger.info("=" * 60)
        logger.info("LOSO: %s (dual thresholds)", model_name)
        logger.info("=" * 60)

        model = all_models[model_name]
        results = run_loso_dual(
            feats_df, feature_cols, model, model_name,
            target_recall=args.target_recall,
            calibrated=do_calibrate,
        )

        for mode in ["detection", "balanced"]:
            r = results[mode]
            tag = f"{model_name}_{mode}"

            # Save per-model per-mode
            r["metrics_df"].to_csv(out_dir / f"loso_metrics_{tag}.csv", index=False)
            if not r["bal_metrics_df"].empty:
                r["bal_metrics_df"].to_csv(out_dir / f"loso_balanced_eval_{tag}.csv", index=False)

            plot_confusion(r["cm"], f"{model_name} — {mode}", out_dir / f"confusion_{tag}.png")
            plot_threshold_hist(r["thresholds"], f"{model_name} — {mode}", out_dir / f"threshold_hist_{tag}.png")

            summary = summarize_mode(model_name, mode, r, args.target_recall)
            all_comparison.append(summary)

            logger.info(
                "[%s/%s] AGG: tn=%d fp=%d fn=%d tp=%d | rec=%.3f spec=%.3f bal=%.3f | thr=%.3f±%.3f",
                model_name, mode,
                summary["agg_tn"], summary["agg_fp"], summary["agg_fn"], summary["agg_tp"],
                summary["mean_recall"], summary["mean_specificity"], summary["mean_balanced_accuracy"],
                summary["mean_threshold"], summary["std_threshold"],
            )

        # PR/ROC uses detection mode probabilities (same probs, just different thresholds)
        plot_pr_roc(results["detection"]["y_true"], results["detection"]["y_prob"],
                    model_name, out_dir / f"pr_roc_{model_name}.png")

        all_detect_metrics[model_name] = results["detection"]["metrics_df"]
        all_balanced_metrics[model_name] = results["balanced"]["metrics_df"]

    # ── Comparison outputs ───────────────────────────────────
    comp_df = pd.DataFrame(all_comparison)
    comp_df.to_csv(out_dir / "loso_metrics_comparison.csv", index=False)

    plot_dual_comparison(all_comparison, out_dir / "dual_comparison.png")

    # Subject diagnostics for detection mode (the one you care about most)
    plot_subject_diagnostics(
        {f"{k} (detect)": v for k, v in all_detect_metrics.items()},
        out_dir / "subject_diagnostics_detection.png",
    )
    plot_subject_diagnostics(
        {f"{k} (balanced)": v for k, v in all_balanced_metrics.items()},
        out_dir / "subject_diagnostics_balanced.png",
    )

    # ── Final summary ────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("FINAL SUMMARY (both operating points)")
    logger.info("=" * 60)
    logger.info("%-10s %-11s | rec    spec   bal_acc | fn     fp     | thr", "MODEL", "MODE")
    logger.info("-" * 75)
    for _, r in comp_df.iterrows():
        logger.info(
            "%-10s %-11s | %.3f  %.3f  %.3f   | %-6d %-6d | %.3f",
            r["model"], r["mode"],
            r["mean_recall"], r["mean_specificity"], r["mean_balanced_accuracy"],
            r["agg_fn"], r["agg_fp"], r["mean_threshold"],
        )
    logger.info("Outputs → %s", out_dir)


if __name__ == "__main__":
    main()