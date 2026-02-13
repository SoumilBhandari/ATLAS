"""
Synthetic TMS-EMG (MEP) Epoch Generator v2
===========================================
Simulates realistic TMS-EMG epochs with configurable preprocessing and analysis.

Architecture improvements over v1:
  - Separated signal generation, preprocessing, analysis, and plotting into clear modules
  - Added spectral analysis tools (FFT visualization, SNR spectrum)
  - Added envelope extraction (rectified + smoothed, sliding RMS)
  - Added phase distortion comparison (filtfilt vs lfilter)
  - Added power-line contamination injection for notch filter demos
  - Improved artifact model (amplifier saturation + recovery)
  - Added polyphase MEP waveform model with realistic motor unit dispersion
  - Trial rejection based on pre-stimulus RMS
  - Proper logging instead of print statements
  - Type-safe throughout with full dataclass validation

Internal convention: ALL voltage arrays stored in Volts (V).
Convert to µV only at plot/export boundaries.

Requires: numpy, scipy, matplotlib
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

try:
    from scipy.signal import (
        butter, sosfiltfilt, sosfilt,   # bandpass: sos form (numerically stable)
        iirnotch, filtfilt, lfilter,    # notch: ba form (iirnotch only returns ba)
    )
except ImportError as e:
    raise ImportError("Requires scipy ≥ 1.1. Install: pip install scipy") from e

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════
#  Constants
# ════════════════════════════════════════════════
V_TO_UV: float = 1e6
UV_TO_V: float = 1e-6

DEFAULT_FS: int = 5_000          # Hz — standard for TMS-EMG
DEFAULT_PRE_MS: float = 100.0    # pre-stimulus window
DEFAULT_POST_MS: float = 100.0   # post-stimulus window
DEFAULT_NOISE_RMS_UV: float = 12.0
DEFAULT_LATENCY_MS: float = 22.0
DEFAULT_ARTIFACT_V: float = 0.003  # 3 mV peak


# ════════════════════════════════════════════════
#  Enums & Profiles
# ════════════════════════════════════════════════
class MuscleTarget(Enum):
    """Supported target muscles with literature-based default profiles."""
    FDI = "first_dorsal_interosseous"
    APB = "abductor_pollicis_brevis"
    TA  = "tibialis_anterior"
    CUSTOM = "custom"


@dataclass(frozen=True)
class MuscleProfile:
    """Literature-based default parameters for a target muscle."""
    latency_ms: float       # typical corticomotor latency
    noise_rms_uv: float     # resting EMG noise floor
    mep_sigma1_ms: float    # rise phase width
    mep_sigma2_ms: float    # fall phase width

    def __post_init__(self):
        if self.latency_ms <= 0:
            raise ValueError(f"latency_ms must be positive, got {self.latency_ms}")
        if self.noise_rms_uv <= 0:
            raise ValueError(f"noise_rms_uv must be positive, got {self.noise_rms_uv}")


MUSCLE_PROFILES: Dict[MuscleTarget, MuscleProfile] = {
    MuscleTarget.FDI: MuscleProfile(latency_ms=22.0, noise_rms_uv=12.0, mep_sigma1_ms=3.5, mep_sigma2_ms=4.5),
    MuscleTarget.APB: MuscleProfile(latency_ms=23.5, noise_rms_uv=14.0, mep_sigma1_ms=3.8, mep_sigma2_ms=4.8),
    MuscleTarget.TA:  MuscleProfile(latency_ms=32.0, noise_rms_uv=18.0, mep_sigma1_ms=5.0, mep_sigma2_ms=6.0),
}


# ════════════════════════════════════════════════
#  Configuration Dataclasses
# ════════════════════════════════════════════════
@dataclass
class PreprocessConfig:
    """
    EMG preprocessing pipeline configuration.

    Pipeline order (each step independently toggleable):
      1. Artifact blanking — replace artifact-contaminated samples before filtering
         (prevents filter ringing on the massive TMS transient)
      2. Bandpass filter — keep 10–500 Hz (physiological EMG band)
      3. Notch filter — remove 50/60 Hz power-line interference + harmonics
      4. Baseline correction — subtract mean of pre-stimulus window

    Why this order matters:
      - Blanking MUST come before filtering: the artifact is orders of magnitude
        larger than EMG. Filtering it without blanking causes severe ringing.
      - Bandpass before notch: reduces broadband noise first, making the notch
        more effective and less likely to distort nearby frequencies.
      - Baseline correction last: filtering can introduce DC offset, so we
        re-center after all frequency-domain operations.
    """
    enabled: bool = True

    # Step 1: Artifact blanking
    do_artifact_blanking: bool = True
    blank_window_ms: Tuple[float, float] = (-1.0, 6.0)
    blank_method: str = "linear"  # "linear", "zero", or "cubic"

    # Step 2: Bandpass
    do_bandpass: bool = True
    bandpass_hz: Tuple[float, float] = (10.0, 500.0)
    bandpass_order: int = 4
    use_zero_phase: bool = True  # filtfilt vs lfilter — educational toggle

    # Step 3: Notch
    do_notch: bool = True
    notch_hz: float = 60.0
    notch_q: float = 30.0
    notch_harmonics: Tuple[int, ...] = (2, 3)  # e.g., 120 Hz, 180 Hz

    # Step 4: Baseline correction
    do_baseline_correction: bool = True
    baseline_window_ms: Tuple[float, float] = (-100.0, -10.0)

    # Trial rejection
    do_trial_rejection: bool = False
    rejection_threshold_uv: float = 50.0  # reject if pre-stim RMS exceeds this
    rejection_window_ms: Tuple[float, float] = (-100.0, -5.0)

    def __post_init__(self):
        lo, hi = self.bandpass_hz
        if lo >= hi:
            raise ValueError(f"Bandpass low ({lo}) must be < high ({hi})")
        if self.blank_method not in ("linear", "zero", "cubic"):
            raise ValueError(f"blank_method must be 'linear', 'zero', or 'cubic', got '{self.blank_method}'")


@dataclass
class ContaminationConfig:
    """
    Optional signal contaminations for educational demos.
    Inject these to practice removing them with preprocessing.
    """
    # Power-line contamination
    add_powerline: bool = False
    powerline_freq_hz: float = 60.0
    powerline_amplitude_uv: float = 15.0
    powerline_harmonics: Tuple[int, ...] = (2, 3)
    powerline_harmonic_decay: float = 0.5  # each harmonic is this × previous

    # Low-frequency drift
    add_drift: bool = False
    drift_freq_hz: float = 1.0
    drift_amplitude_uv: float = 40.0

    # High-frequency amplifier noise
    add_hf_noise: bool = False
    hf_noise_freq_hz: float = 1000.0
    hf_noise_rms_uv: float = 8.0


@dataclass
class EpochConfig:
    """Full configuration for generating a single synthetic TMS-EMG epoch."""
    # Stimulation parameters
    intensity_pct: float = 50.0
    true_threshold_pct: float = 50.0
    true_slope: float = 0.25

    # EMG noise
    noise_rms_uv: float = DEFAULT_NOISE_RMS_UV

    # MEP parameters
    latency_ms: float = DEFAULT_LATENCY_MS
    latency_jitter_ms: float = 0.8
    mep_sigma1_ms: float = 3.5
    mep_sigma2_ms: float = 4.5
    mep_lobe_separation_ms: float = 3.0
    mep_max_p2p_uv: float = 5000.0
    mep_n_phases: int = 3  # polyphasic complexity (2 = biphasic, 3+ = polyphasic)

    # Artifact parameters
    artifact_amplitude_V: float = DEFAULT_ARTIFACT_V
    artifact_bipolar: bool = True
    artifact_ring_freq_hz: float = 2000.0
    artifact_ring_tau_s: float = 0.004
    artifact_saturate: bool = True
    artifact_saturation_V: float = 0.005  # amplifier rail voltage

    # Pre-innervation (voluntary contraction before TMS)
    pre_innervation: bool = False
    pre_innervation_rms_uv: float = 25.0
    pre_innervation_freq_hz: float = 35.0

    # Cortical silent period
    silent_period: bool = True
    silent_period_duration_ms: float = 30.0
    silent_period_suppression: float = 0.85

    # Timing
    sampling_rate: int = DEFAULT_FS
    pre_ms: float = DEFAULT_PRE_MS
    post_ms: float = DEFAULT_POST_MS

    # Sub-configs
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    contamination: ContaminationConfig = field(default_factory=ContaminationConfig)

    seed: Optional[int] = None

    def __post_init__(self):
        if self.sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {self.sampling_rate}")
        nyquist = self.sampling_rate / 2.0
        bp_hi = self.preprocess.bandpass_hz[1]
        if bp_hi >= nyquist:
            raise ValueError(
                f"Bandpass high ({bp_hi} Hz) must be below Nyquist ({nyquist} Hz) "
                f"for fs={self.sampling_rate} Hz. Either lower bandpass_hz or increase sampling_rate."
            )

    @classmethod
    def from_muscle(cls, muscle: MuscleTarget, intensity_pct: float = 50.0, **overrides) -> "EpochConfig":
        profile = MUSCLE_PROFILES.get(muscle)
        if profile is None:
            raise ValueError(f"No profile for {muscle}. Use MuscleTarget.CUSTOM with explicit params.")
        defaults = dict(
            intensity_pct=intensity_pct,
            latency_ms=profile.latency_ms,
            noise_rms_uv=profile.noise_rms_uv,
            mep_sigma1_ms=profile.mep_sigma1_ms,
            mep_sigma2_ms=profile.mep_sigma2_ms,
        )
        defaults.update(overrides)
        return cls(**defaults)


# ════════════════════════════════════════════════
#  Result Dataclass
# ════════════════════════════════════════════════
@dataclass
class TrialResult:
    """Stores all metadata and measurements from a single simulated trial."""
    intensity_pct: float
    true_threshold_pct: float
    true_slope: float

    response_probability: float
    response_occurred: bool

    mep_p2p_uv_true: float
    latency_ms_true: Optional[float]

    ptp_uv_raw: float
    ptp_uv_processed: float

    snr_db_true: float

    pre_innervation: bool
    pre_stim_rms_uv: float     # measured pre-stimulus RMS
    trial_rejected: bool       # whether trial would be rejected
    noise_rms_uv_target: float
    artifact_peak_V: float
    fs: int


# ════════════════════════════════════════════════
#  Signal Processing Utilities
# ════════════════════════════════════════════════
def _rms(x: np.ndarray) -> float:
    """Root mean square of a signal."""
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _scale_to_rms(signal: np.ndarray, target_rms: float) -> np.ndarray:
    """Rescale signal to have a specific RMS amplitude."""
    current = _rms(signal)
    if current < 1e-18:
        return signal
    return signal * (target_rms / current)


# ────────────────────────────────────────────────
#  Core DSP Functions
# ────────────────────────────────────────────────
def bandpass_filter(
    x: np.ndarray,
    fs: int,
    low_hz: float = 10.0,
    high_hz: float = 500.0,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    Butterworth bandpass filter using second-order sections (SOS) for
    numerical stability.

    Parameters
    ----------
    x : signal
    fs : sampling rate (Hz)
    low_hz, high_hz : passband edges
    order : filter order (applied twice if zero_phase → effective order 2×)
    zero_phase : if True, uses sosfiltfilt (no phase distortion, non-causal)
                 if False, uses sosfilt (causal, introduces phase shift)

    Why SOS?
      High-order IIR filters are numerically unstable in transfer-function
      (b, a) form. SOS cascades second-order sections and avoids this.
      Always prefer SOS for Butterworth order > 2.
    """
    nyq = fs / 2.0
    lo = max(0.1, low_hz) / nyq
    hi = min(high_hz, nyq - 1.0) / nyq
    if not (0 < lo < hi < 1):
        raise ValueError(f"Invalid bandpass: low={low_hz}, high={high_hz}, fs={fs}")

    sos = butter(order, [lo, hi], btype="bandpass", output="sos")

    if zero_phase:
        return sosfiltfilt(sos, x)
    else:
        return sosfilt(sos, x)


def notch_filter(
    x: np.ndarray,
    fs: int,
    freq_hz: float = 60.0,
    q: float = 30.0,
    zero_phase: bool = True,
) -> np.ndarray:
    """
    IIR notch (band-reject) filter to remove a specific frequency.

    Typical use: removing 50/60 Hz power-line interference.

    Parameters
    ----------
    q : quality factor. Higher Q = narrower notch.
        Q=30 removes ~2 Hz band around center. Q=10 removes ~6 Hz band.
    """
    nyq = fs / 2.0
    if freq_hz >= nyq:
        warnings.warn(f"Notch freq {freq_hz} Hz ≥ Nyquist {nyq} Hz — skipping.")
        return x.copy()

    b, a = iirnotch(w0=freq_hz, Q=q, fs=fs)
    if zero_phase:
        return filtfilt(b, a, x)
    else:
        return lfilter(b, a, x)


def baseline_correct(
    time_ms: np.ndarray,
    v: np.ndarray,
    window_ms: Tuple[float, float],
) -> np.ndarray:
    """Subtract mean voltage in baseline window."""
    mask = (time_ms >= window_ms[0]) & (time_ms <= window_ms[1])
    if not np.any(mask):
        warnings.warn("Baseline window contains no samples — skipping correction.")
        return v.copy()
    return v - float(np.mean(v[mask]))


def blank_artifact(
    time_ms: np.ndarray,
    voltage_V: np.ndarray,
    blank_window_ms: Tuple[float, float] = (-1.0, 6.0),
    method: str = "linear",
) -> np.ndarray:
    """
    Replace artifact-contaminated samples.

    Methods:
      "zero"   — replace with zeros (simple but creates a discontinuity)
      "linear" — linear interpolation between window edges (preserves continuity)
      "cubic"  — cubic interpolation using 4 anchor points (smoother)

    Why blank before filtering?
      The TMS artifact is ~1000× larger than EMG. If you bandpass first,
      the filter rings violently on the artifact edges, spreading contamination
      across 10–20 ms of otherwise clean data. Blanking first removes the
      extreme transient so the filter has nothing to ring on.
    """
    v = voltage_V.copy()
    mask = (time_ms >= blank_window_ms[0]) & (time_ms <= blank_window_ms[1])
    if not np.any(mask):
        return v

    idx = np.where(mask)[0]
    i0, i1 = idx[0], idx[-1]
    left = max(i0 - 1, 0)
    right = min(i1 + 1, len(v) - 1)

    if method == "zero":
        v[idx] = 0.0
    elif method == "linear":
        v[idx] = np.linspace(v[left], v[right], len(idx))
    elif method == "cubic":
        # Use 2 points on each side for cubic spline
        left2 = max(left - 1, 0)
        right2 = min(right + 1, len(v) - 1)
        x_anchor = np.array([left2, left, right, right2], dtype=float)
        y_anchor = np.array([v[left2], v[left], v[right], v[right2]])
        v[idx] = np.interp(idx.astype(float), x_anchor, y_anchor)
    else:
        raise ValueError(f"Unknown blank method: {method}")

    return v


def compute_envelope(
    signal_V: np.ndarray,
    fs: int,
    method: str = "rms",
    window_ms: float = 20.0,
) -> np.ndarray:
    """
    Extract the amplitude envelope of an EMG signal.

    Methods:
      "rectify" — full-wave rectify + low-pass smooth
      "rms"     — sliding-window RMS (more standard for EMG)

    The envelope represents the time-varying amplitude of muscle activation.
    Useful for: detecting voluntary contraction, silent period analysis,
    pre-stimulus EMG monitoring.
    """
    window_samples = max(1, int(round(window_ms / 1000.0 * fs)))

    if method == "rms":
        squared = np.square(signal_V)
        # Cumulative sum trick for efficient sliding window
        cumsum = np.cumsum(np.insert(squared, 0, 0))
        # Pad edges to maintain length
        envelope = np.zeros_like(signal_V)
        half_w = window_samples // 2
        for i in range(len(signal_V)):
            lo = max(0, i - half_w)
            hi = min(len(signal_V), i + half_w + 1)
            envelope[i] = np.sqrt((cumsum[hi] - cumsum[lo]) / (hi - lo))
        return envelope

    elif method == "rectify":
        rectified = np.abs(signal_V)
        # Smooth with low-pass filter
        cutoff_hz = 1000.0 / window_ms  # window_ms → equivalent cutoff
        cutoff_hz = min(cutoff_hz, fs / 2.0 - 1.0)
        if cutoff_hz > 0.5:
            sos = butter(2, cutoff_hz / (fs / 2.0), btype="low", output="sos")
            return sosfiltfilt(sos, rectified)
        return rectified

    else:
        raise ValueError(f"Unknown envelope method: {method}")


def compute_spectrum(
    signal_V: np.ndarray,
    fs: int,
    window: str = "hann",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute single-sided amplitude spectrum via FFT.

    Returns (frequencies_hz, magnitude_uv).
    Applies a window function to reduce spectral leakage.
    """
    n = len(signal_V)

    # Apply window
    if window == "hann":
        w = np.hanning(n)
    elif window == "hamming":
        w = np.hamming(n)
    elif window == "none":
        w = np.ones(n)
    else:
        w = np.hanning(n)

    windowed = signal_V * w

    fft_vals = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitude_V = 2.0 * np.abs(fft_vals) / n
    magnitude_uv = magnitude_V * V_TO_UV

    return freqs, magnitude_uv


# ════════════════════════════════════════════════
#  Preprocessing Pipeline
# ════════════════════════════════════════════════
def preprocess_epoch(
    time_ms: np.ndarray,
    voltage_V: np.ndarray,
    fs: int,
    pcfg: PreprocessConfig,
) -> np.ndarray:
    """
    Apply the full preprocessing pipeline to a single epoch.

    Order is critical — see PreprocessConfig docstring for rationale.
    """
    if not pcfg.enabled:
        return voltage_V.copy()

    v = voltage_V.copy()

    # 1) Artifact blanking
    if pcfg.do_artifact_blanking:
        v = blank_artifact(time_ms, v, pcfg.blank_window_ms, method=pcfg.blank_method)

    # 2) Bandpass
    if pcfg.do_bandpass:
        lo, hi = pcfg.bandpass_hz
        v = bandpass_filter(v, fs, lo, hi, order=pcfg.bandpass_order,
                            zero_phase=pcfg.use_zero_phase)

    # 3) Notch + harmonics
    if pcfg.do_notch:
        v = notch_filter(v, fs, pcfg.notch_hz, pcfg.notch_q,
                         zero_phase=pcfg.use_zero_phase)
        for h in pcfg.notch_harmonics:
            v = notch_filter(v, fs, pcfg.notch_hz * h, pcfg.notch_q,
                             zero_phase=pcfg.use_zero_phase)

    # 4) Baseline correction
    if pcfg.do_baseline_correction:
        v = baseline_correct(time_ms, v, pcfg.baseline_window_ms)

    return v


def check_trial_rejection(
    time_ms: np.ndarray,
    voltage_V: np.ndarray,
    pcfg: PreprocessConfig,
) -> Tuple[bool, float]:
    """
    Check if a trial should be rejected based on pre-stimulus EMG.

    High pre-stimulus RMS indicates voluntary muscle contraction,
    which contaminates MEP amplitude measurement.

    Returns (rejected: bool, pre_stim_rms_uv: float).
    """
    if not pcfg.do_trial_rejection:
        # Still compute RMS for reporting even if not rejecting
        mask = (time_ms >= pcfg.rejection_window_ms[0]) & (time_ms <= pcfg.rejection_window_ms[1])
        pre_rms_uv = _rms(voltage_V[mask]) * V_TO_UV if np.any(mask) else 0.0
        return False, pre_rms_uv

    mask = (time_ms >= pcfg.rejection_window_ms[0]) & (time_ms <= pcfg.rejection_window_ms[1])
    if not np.any(mask):
        return False, 0.0

    pre_rms_uv = _rms(voltage_V[mask]) * V_TO_UV
    rejected = pre_rms_uv > pcfg.rejection_threshold_uv
    return rejected, pre_rms_uv


# ════════════════════════════════════════════════
#  Signal Component Builders
# ════════════════════════════════════════════════
def _make_time_base(cfg: EpochConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Generate time vectors in seconds and milliseconds."""
    dt = 1.0 / cfg.sampling_rate
    t_s = np.arange(-cfg.pre_ms / 1000.0, cfg.post_ms / 1000.0 + dt, dt)
    return t_s, t_s * 1000.0


def _make_noise(t_s: np.ndarray, cfg: EpochConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Generate bandpass-filtered noise simulating resting EMG.

    Real resting EMG is not white noise — it has a characteristic spectral
    shape concentrated in the 20–200 Hz range with a peak around 50–80 Hz.
    We approximate this by bandpass-filtering white Gaussian noise.
    """
    # Generate enough extra samples to avoid filter edge effects
    pad = cfg.sampling_rate  # 1 second padding
    white = rng.normal(0.0, 1.0, size=len(t_s) + 2 * pad)
    bp = bandpass_filter(white, cfg.sampling_rate, 10.0, 500.0, order=4)
    # Trim padding
    bp = bp[pad:pad + len(t_s)]
    return _scale_to_rms(bp, cfg.noise_rms_uv * UV_TO_V)


def _make_contamination(t_s: np.ndarray, cfg: EpochConfig) -> np.ndarray:
    """
    Inject deliberate signal contaminations for educational purposes.
    Allows practicing removal with filters.
    """
    cc = cfg.contamination
    out = np.zeros_like(t_s)

    # Power-line interference (50/60 Hz + harmonics)
    if cc.add_powerline:
        amp = cc.powerline_amplitude_uv * UV_TO_V
        out += amp * np.sin(2.0 * np.pi * cc.powerline_freq_hz * t_s)
        decay = 1.0
        for h in cc.powerline_harmonics:
            decay *= cc.powerline_harmonic_decay
            out += amp * decay * np.sin(2.0 * np.pi * cc.powerline_freq_hz * h * t_s)

    # Low-frequency baseline drift
    if cc.add_drift:
        out += cc.drift_amplitude_uv * UV_TO_V * np.sin(2.0 * np.pi * cc.drift_freq_hz * t_s)

    # High-frequency noise
    if cc.add_hf_noise:
        hf = np.sin(2.0 * np.pi * cc.hf_noise_freq_hz * t_s)
        out += _scale_to_rms(hf, cc.hf_noise_rms_uv * UV_TO_V)

    return out


def _make_pre_innervation(
    t_s: np.ndarray,
    time_ms: np.ndarray,
    cfg: EpochConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate voluntary muscle contraction before TMS pulse."""
    pad = cfg.sampling_rate
    extra_white = rng.normal(0.0, 1.0, size=len(t_s) + 2 * pad)
    extra_bp = bandpass_filter(extra_white, cfg.sampling_rate, 20.0, 250.0, order=2)
    extra_bp = extra_bp[pad:pad + len(t_s)]
    extra = _scale_to_rms(extra_bp, cfg.pre_innervation_rms_uv * UV_TO_V)

    mask = (time_ms >= -cfg.pre_ms) & (time_ms <= -10.0)
    out = np.zeros_like(t_s)
    out[mask] = extra[mask]
    # Add a tonic motor unit firing component
    out[mask] += 8e-6 * np.sin(2.0 * np.pi * cfg.pre_innervation_freq_hz * t_s[mask])
    return out


def _make_artifact(t_s: np.ndarray, time_ms: np.ndarray, cfg: EpochConfig) -> np.ndarray:
    """
    Simulate TMS stimulus artifact.

    Components:
      1. Sharp onset spike at t=0 (electromagnetic coupling)
      2. Exponential decay (amplifier recovery)
      3. Bipolar rebound (charge redistribution at electrode-skin interface)
      4. Damped ringing (amplifier oscillation / cable resonance)
      5. Optional amplifier saturation (clipping at rail voltage)
    """
    artifact = np.zeros_like(t_s)
    post_mask = time_ms >= 0.0
    tt = time_ms[post_mask] / 1000.0

    amp = cfg.artifact_amplitude_V
    tau_fast = 0.0015
    tau_slow = 0.003

    # Positive lobe (exponential decay)
    artifact[post_mask] += amp * np.exp(-tt / tau_fast)

    # Negative rebound (bipolar)
    if cfg.artifact_bipolar:
        artifact[post_mask] -= 0.6 * amp * np.exp(-tt / tau_slow) * np.sin(np.pi * tt / 0.004)

    # Damped ringing
    artifact[post_mask] += (amp * 0.35) * np.exp(-tt / cfg.artifact_ring_tau_s) * np.sin(
        2.0 * np.pi * cfg.artifact_ring_freq_hz * tt
    )

    # Sharp onset spike
    idx0 = int(np.argmin(np.abs(time_ms)))
    artifact[idx0] += amp * 1.5

    # Amplifier saturation (clipping)
    if cfg.artifact_saturate:
        rail = cfg.artifact_saturation_V
        artifact = np.clip(artifact, -rail, rail)

    return artifact


def _make_mep(
    t_s: np.ndarray,
    time_ms: np.ndarray,
    cfg: EpochConfig,
    rng: np.random.Generator,
    intensity_gain: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Generate a motor evoked potential (MEP) waveform.

    Uses a polyphasic model: sum of Gaussian-windowed lobes with alternating
    polarity and decreasing amplitude. More realistic than simple biphasic.

    The polyphasic nature reflects temporal dispersion of descending volleys
    activating motor units with slightly different latencies.
    """
    # Amplitude model
    base_uv = 60.0 + 35.0 * intensity_gain
    variability = float(np.exp(rng.normal(0.0, 0.35)))
    p2p_uv = float(min(base_uv * variability, cfg.mep_max_p2p_uv))
    p2p_V = p2p_uv * UV_TO_V

    # Latency with jitter
    latency_ms = float(cfg.latency_ms + rng.normal(0.0, cfg.latency_jitter_ms))
    t_center = latency_ms / 1000.0

    s1 = cfg.mep_sigma1_ms / 1000.0
    s2 = cfg.mep_sigma2_ms / 1000.0
    d = cfg.mep_lobe_separation_ms / 1000.0

    # Build polyphasic waveform
    n_phases = max(2, cfg.mep_n_phases)
    waveform = np.zeros_like(t_s)
    for i in range(n_phases):
        sigma = s1 if i == 0 else s2 * (1.0 + 0.1 * i)  # later phases slightly broader
        center = t_center + i * d
        polarity = 1.0 if i % 2 == 0 else -1.0
        amplitude = polarity * (0.8 ** i)  # decreasing amplitude
        waveform += amplitude * np.exp(-0.5 * ((t_s - center) / sigma) ** 2)

    # Remove any DC offset relative to baseline
    bl_mask = (time_ms >= -cfg.pre_ms) & (time_ms <= -10.0)
    if np.any(bl_mask):
        waveform -= float(np.mean(waveform[bl_mask]))

    # Scale to target peak-to-peak
    w_range = float(np.max(waveform) - np.min(waveform))
    if w_range > 1e-18:
        waveform *= p2p_V / w_range

    return waveform, p2p_uv, latency_ms


def _apply_silent_period(
    voltage: np.ndarray,
    time_ms: np.ndarray,
    cfg: EpochConfig,
    latency_ms: float,
) -> np.ndarray:
    """
    Apply cortical silent period (CSP) — suppression of EMG after MEP.

    Uses a cosine-tapered suppression envelope for smooth onset/offset.
    """
    csp_start = latency_ms + 15.0
    csp_end = csp_start + cfg.silent_period_duration_ms

    mask = (time_ms >= csp_start) & (time_ms <= csp_end)
    if not np.any(mask):
        return voltage

    t_local = (time_ms[mask] - csp_start) / cfg.silent_period_duration_ms
    suppression = cfg.silent_period_suppression * 0.5 * (1.0 + np.cos(np.pi * t_local))

    voltage = voltage.copy()
    voltage[mask] *= (1.0 - suppression)
    return voltage


# ════════════════════════════════════════════════
#  Analysis
# ════════════════════════════════════════════════
def compute_p2p_uv(
    time_ms: np.ndarray,
    voltage_V: np.ndarray,
    window_ms: Tuple[float, float] = (15.0, 45.0),
) -> float:
    """Peak-to-peak amplitude in µV within a time window."""
    mask = (time_ms >= window_ms[0]) & (time_ms <= window_ms[1])
    if not np.any(mask):
        return 0.0
    seg = voltage_V[mask]
    return float((np.max(seg) - np.min(seg)) * V_TO_UV)


def compute_snr_db(
    time_ms: np.ndarray,
    voltage_V: np.ndarray,
    signal_window_ms: Tuple[float, float] = (15.0, 45.0),
    noise_window_ms: Tuple[float, float] = (-100.0, -5.0),
) -> float:
    """
    Empirical SNR in dB from a single epoch.
    Compares RMS in MEP window to RMS in pre-stimulus baseline.
    """
    sig_mask = (time_ms >= signal_window_ms[0]) & (time_ms <= signal_window_ms[1])
    noise_mask = (time_ms >= noise_window_ms[0]) & (time_ms <= noise_window_ms[1])

    sig_rms = _rms(voltage_V[sig_mask]) if np.any(sig_mask) else 0.0
    noise_rms = _rms(voltage_V[noise_mask]) if np.any(noise_mask) else 1e-18

    if noise_rms < 1e-18:
        return float("inf") if sig_rms > 0 else 0.0

    return 20.0 * math.log10(sig_rms / noise_rms)


# ════════════════════════════════════════════════
#  Main Simulator
# ════════════════════════════════════════════════
def simulate_trial(cfg: EpochConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TrialResult]:
    """
    Simulate a single TMS-EMG trial.

    Returns
    -------
    time_ms : time vector (ms)
    voltage_raw_V : raw signal (V)
    voltage_processed_V : preprocessed signal (V)
    result : TrialResult with all metadata and measurements
    """
    rng = np.random.default_rng(cfg.seed)
    t_s, time_ms = _make_time_base(cfg)

    # 1) Baseline EMG noise
    voltage = _make_noise(t_s, cfg, rng)

    # 2) Deliberate contaminations (for educational demos)
    voltage += _make_contamination(t_s, cfg)

    # 3) Optional pre-innervation
    if cfg.pre_innervation:
        voltage += _make_pre_innervation(t_s, time_ms, cfg, rng)

    # 4) Stimulus artifact
    artifact = _make_artifact(t_s, time_ms, cfg)
    voltage += artifact

    # 5) Response probability (sigmoid model)
    p_resp = _sigmoid(cfg.true_slope * (cfg.intensity_pct - cfg.true_threshold_pct))
    response_occurs = bool(rng.random() < p_resp)

    # 6) MEP (if response occurred)
    mep_p2p_uv_true = 0.0
    latency_ms_true: Optional[float] = None
    if response_occurs:
        intensity_gain = max(0.0, cfg.intensity_pct - cfg.true_threshold_pct)
        mep_wave, mep_p2p_uv_true, latency_ms_true = _make_mep(t_s, time_ms, cfg, rng, intensity_gain)
        voltage += mep_wave
        if cfg.silent_period and latency_ms_true is not None:
            voltage = _apply_silent_period(voltage, time_ms, cfg, latency_ms_true)

    voltage_raw = voltage

    # 7) Trial rejection check (on raw signal)
    rejected, pre_stim_rms = check_trial_rejection(time_ms, voltage_raw, cfg.preprocess)

    # 8) Preprocessing
    voltage_proc = preprocess_epoch(time_ms, voltage_raw, cfg.sampling_rate, cfg.preprocess)

    # 9) Measurements
    ptp_uv_raw = compute_p2p_uv(time_ms, voltage_raw)
    ptp_uv_proc = compute_p2p_uv(time_ms, voltage_proc)

    # True SNR (from known injected components)
    noise_floor = cfg.noise_rms_uv * UV_TO_V
    snr_db_true = (
        10.0 * math.log10(max(mep_p2p_uv_true * UV_TO_V, 1e-18) / noise_floor)
        if response_occurs else float("-inf")
    )

    result = TrialResult(
        intensity_pct=cfg.intensity_pct,
        true_threshold_pct=cfg.true_threshold_pct,
        true_slope=cfg.true_slope,
        response_probability=p_resp,
        response_occurred=response_occurs,
        mep_p2p_uv_true=mep_p2p_uv_true,
        latency_ms_true=latency_ms_true,
        ptp_uv_raw=ptp_uv_raw,
        ptp_uv_processed=ptp_uv_proc,
        snr_db_true=snr_db_true,
        pre_innervation=cfg.pre_innervation,
        pre_stim_rms_uv=pre_stim_rms,
        trial_rejected=rejected,
        noise_rms_uv_target=cfg.noise_rms_uv,
        artifact_peak_V=float(np.max(np.abs(artifact))),
        fs=cfg.sampling_rate,
    )

    return time_ms, voltage_raw, voltage_proc, result


# ════════════════════════════════════════════════
#  Recruitment Curve
# ════════════════════════════════════════════════
@dataclass
class RecruitmentCurve:
    intensities: np.ndarray
    ptp_uv_raw: np.ndarray
    ptp_uv_processed: np.ndarray
    response_occurred: np.ndarray
    response_probability: np.ndarray
    pre_stim_rms_uv: np.ndarray
    trial_rejected: np.ndarray
    results: List[TrialResult]

    def estimate_threshold(self, criterion_uv: float = 50.0, use_processed: bool = True) -> Optional[float]:
        """Estimate motor threshold as lowest intensity with ≥50% supra-threshold responses."""
        y = self.ptp_uv_processed if use_processed else self.ptp_uv_raw
        # Optionally exclude rejected trials
        valid = ~self.trial_rejected
        unique_ints = np.unique(self.intensities[valid])
        for intensity in sorted(unique_ints):
            mask = (self.intensities == intensity) & valid
            if not np.any(mask):
                continue
            frac = np.mean(y[mask] >= criterion_uv)
            if frac >= 0.5:
                return float(intensity)
        return None


def simulate_recruitment_curve(
    intensities: Sequence[float],
    n_trials_per: int = 10,
    base_seed: int = 0,
    **epoch_kwargs,
) -> RecruitmentCurve:
    """Simulate a full recruitment curve across multiple intensities."""
    all_ints: List[float] = []
    all_raw: List[float] = []
    all_proc: List[float] = []
    all_resp: List[bool] = []
    all_prob: List[float] = []
    all_rms: List[float] = []
    all_rej: List[bool] = []
    all_results: List[TrialResult] = []

    trial_idx = 0
    for intensity in intensities:
        for _ in range(n_trials_per):
            cfg = EpochConfig(intensity_pct=float(intensity), seed=base_seed + trial_idx, **epoch_kwargs)
            _, _, _, res = simulate_trial(cfg)
            all_ints.append(res.intensity_pct)
            all_raw.append(res.ptp_uv_raw)
            all_proc.append(res.ptp_uv_processed)
            all_resp.append(res.response_occurred)
            all_prob.append(res.response_probability)
            all_rms.append(res.pre_stim_rms_uv)
            all_rej.append(res.trial_rejected)
            all_results.append(res)
            trial_idx += 1

    return RecruitmentCurve(
        intensities=np.array(all_ints),
        ptp_uv_raw=np.array(all_raw),
        ptp_uv_processed=np.array(all_proc),
        response_occurred=np.array(all_resp, dtype=bool),
        response_probability=np.array(all_prob),
        pre_stim_rms_uv=np.array(all_rms),
        trial_rejected=np.array(all_rej, dtype=bool),
        results=all_results,
    )


# ════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════
def _epoch_title(res: TrialResult) -> str:
    rej_str = " [REJECTED]" if res.trial_rejected else ""
    return (
        f"Intensity {res.intensity_pct:.0f}% | resp={res.response_occurred} | "
        f"true p2p={res.mep_p2p_uv_true:.0f} µV | proc p2p={res.ptp_uv_processed:.0f} µV | "
        f"SNR={res.snr_db_true:.1f} dB | pre-RMS={res.pre_stim_rms_uv:.1f} µV{rej_str}"
    )


def plot_epoch_dual(
    time_ms: np.ndarray,
    raw_V: np.ndarray,
    proc_V: np.ndarray,
    res: TrialResult,
    mep_xlim_ms: Tuple[float, float] = (5.0, 60.0),
    analysis_window_ms: Tuple[float, float] = (15.0, 45.0),
    show: bool = False,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Plot raw full epoch (top) and processed MEP zoom (bottom)."""
    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(12, 7),
                                            gridspec_kw={"height_ratios": [1, 1.2]})

    # Top: RAW full epoch
    ax_full.plot(time_ms, raw_V * V_TO_UV, linewidth=0.6, color="#2c7bb6")
    ax_full.axvline(0.0, color="gray", linestyle="--", linewidth=0.7, label="TMS pulse")
    if res.latency_ms_true is not None:
        ax_full.axvline(res.latency_ms_true, color="red", linestyle=":", linewidth=0.8,
                       label=f"MEP onset ({res.latency_ms_true:.1f} ms)")
    ax_full.axvspan(mep_xlim_ms[0], mep_xlim_ms[1], alpha=0.08, color="blue", label="zoom region")
    ax_full.set_title("RAW epoch (artifact dominates scale)", fontsize=10, fontweight="bold")
    ax_full.set_ylabel("Voltage (µV)")
    ax_full.legend(fontsize=7, loc="upper right")
    ax_full.grid(True, alpha=0.2)

    # Bottom: PROCESSED zoom
    ax_zoom.plot(time_ms, proc_V * V_TO_UV, linewidth=0.9, color="#d7191c")
    ax_zoom.axvline(0.0, color="gray", linestyle="--", linewidth=0.7, label="TMS pulse")
    if res.latency_ms_true is not None:
        ax_zoom.axvline(res.latency_ms_true, color="red", linestyle=":", linewidth=0.8,
                       label=f"MEP onset ({res.latency_ms_true:.1f} ms)")
    ax_zoom.axvspan(analysis_window_ms[0], analysis_window_ms[1], alpha=0.10, color="green",
                   label=f"PTP window ({analysis_window_ms[0]:.0f}–{analysis_window_ms[1]:.0f} ms)")
    ax_zoom.set_xlim(mep_xlim_ms)
    ax_zoom.set_title("PROCESSED epoch (blanked → bandpass → notch → baseline)", fontsize=10, fontweight="bold")
    ax_zoom.set_xlabel("Time (ms)")
    ax_zoom.set_ylabel("Voltage (µV)")
    ax_zoom.legend(fontsize=7, loc="upper right")
    ax_zoom.grid(True, alpha=0.2)

    # Auto-scale y on zoom
    m = (time_ms >= mep_xlim_ms[0]) & (time_ms <= mep_xlim_ms[1])
    seg = proc_V[m] * V_TO_UV
    if seg.size > 0:
        margin = max(20.0, 0.15 * (np.max(seg) - np.min(seg)))
        ax_zoom.set_ylim(np.min(seg) - margin, np.max(seg) + margin)

    fig.suptitle(_epoch_title(res), fontsize=9, y=0.99)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, (ax_full, ax_zoom)


def plot_spectrum(
    time_ms: np.ndarray,
    raw_V: np.ndarray,
    proc_V: np.ndarray,
    fs: int,
    freq_xlim: Tuple[float, float] = (0.0, 600.0),
    show: bool = False,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """
    Plot frequency spectra of raw vs processed signals.
    Essential for building frequency-domain intuition.
    """
    fig, (ax_raw, ax_proc) = plt.subplots(2, 1, figsize=(12, 6))

    freqs_raw, mag_raw = compute_spectrum(raw_V, fs)
    freqs_proc, mag_proc = compute_spectrum(proc_V, fs)

    # Raw spectrum
    ax_raw.semilogy(freqs_raw, mag_raw, linewidth=0.6, color="#2c7bb6")
    ax_raw.axvline(10, color="green", linestyle="--", alpha=0.5, label="HP cutoff (10 Hz)")
    ax_raw.axvline(500, color="orange", linestyle="--", alpha=0.5, label="LP cutoff (500 Hz)")
    ax_raw.axvline(60, color="red", linestyle=":", alpha=0.5, label="60 Hz line noise")
    ax_raw.set_xlim(freq_xlim)
    ax_raw.set_title("RAW spectrum", fontsize=10, fontweight="bold")
    ax_raw.set_ylabel("Magnitude (µV)")
    ax_raw.legend(fontsize=7)
    ax_raw.grid(True, alpha=0.2)

    # Processed spectrum
    ax_proc.semilogy(freqs_proc, mag_proc, linewidth=0.6, color="#d7191c")
    ax_proc.axvline(10, color="green", linestyle="--", alpha=0.5, label="HP cutoff (10 Hz)")
    ax_proc.axvline(500, color="orange", linestyle="--", alpha=0.5, label="LP cutoff (500 Hz)")
    ax_proc.axvline(60, color="red", linestyle=":", alpha=0.5, label="60 Hz line noise")
    ax_proc.set_xlim(freq_xlim)
    ax_proc.set_title("PROCESSED spectrum", fontsize=10, fontweight="bold")
    ax_proc.set_xlabel("Frequency (Hz)")
    ax_proc.set_ylabel("Magnitude (µV)")
    ax_proc.legend(fontsize=7)
    ax_proc.grid(True, alpha=0.2)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, (ax_raw, ax_proc)


def plot_phase_comparison(
    time_ms: np.ndarray,
    voltage_V: np.ndarray,
    fs: int,
    bandpass_hz: Tuple[float, float] = (10.0, 500.0),
    zoom_ms: Tuple[float, float] = (15.0, 50.0),
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """
    Compare zero-phase (filtfilt) vs causal (lfilter) filtering.
    Demonstrates how causal filtering shifts MEP latency and distorts waveform shape.
    """
    # First blank the artifact so we're comparing filter effects, not artifact effects
    blanked = blank_artifact(time_ms, voltage_V, (-1.0, 6.0), method="linear")

    v_zerophase = bandpass_filter(blanked, fs, *bandpass_hz, zero_phase=True)
    v_causal = bandpass_filter(blanked, fs, *bandpass_hz, zero_phase=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    mask = (time_ms >= zoom_ms[0]) & (time_ms <= zoom_ms[1])
    t = time_ms[mask]

    ax.plot(t, blanked[mask] * V_TO_UV, linewidth=0.5, alpha=0.5, color="gray", label="blanked (no filter)")
    ax.plot(t, v_zerophase[mask] * V_TO_UV, linewidth=1.2, color="#2c7bb6", label="filtfilt (zero-phase)")
    ax.plot(t, v_causal[mask] * V_TO_UV, linewidth=1.2, color="#d7191c", linestyle="--", label="lfilter (causal)")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (µV)")
    ax.set_title("Phase Distortion: filtfilt vs lfilter\n"
                 "Notice the latency shift and waveform distortion with causal filtering",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_envelope(
    time_ms: np.ndarray,
    voltage_V: np.ndarray,
    fs: int,
    window_ms: float = 20.0,
    zoom_ms: Optional[Tuple[float, float]] = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot EMG signal with its amplitude envelope overlay."""
    env = compute_envelope(voltage_V, fs, method="rms", window_ms=window_ms)

    fig, ax = plt.subplots(figsize=(12, 4))

    if zoom_ms is not None:
        mask = (time_ms >= zoom_ms[0]) & (time_ms <= zoom_ms[1])
    else:
        mask = np.ones(len(time_ms), dtype=bool)

    ax.plot(time_ms[mask], voltage_V[mask] * V_TO_UV, linewidth=0.4, alpha=0.6, color="#2c7bb6", label="EMG")
    ax.plot(time_ms[mask], env[mask] * V_TO_UV, linewidth=1.5, color="#d7191c",
            label=f"RMS envelope ({window_ms:.0f} ms window)")
    ax.plot(time_ms[mask], -env[mask] * V_TO_UV, linewidth=1.5, color="#d7191c", alpha=0.5)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (µV)")
    ax.set_title("EMG Envelope Extraction (sliding RMS)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_overlay(
    epochs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, TrialResult]],
    xlim: Tuple[float, float] = (-50.0, 80.0),
    mep_xlim: Tuple[float, float] = (5.0, 60.0),
    show: bool = False,
) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Overlay multiple epochs colored by intensity."""
    fig, (ax_raw, ax_proc) = plt.subplots(2, 1, figsize=(12, 7),
                                           gridspec_kw={"height_ratios": [1, 1.2]})

    cmap = plt.cm.viridis
    n = len(epochs)
    for i, (t_ms, raw, proc, res) in enumerate(epochs):
        color = cmap(i / max(n - 1, 1))
        ax_raw.plot(t_ms, raw * V_TO_UV, alpha=0.6, linewidth=0.6, color=color,
                   label=f"{res.intensity_pct:.0f}%")
        ax_proc.plot(t_ms, proc * V_TO_UV, alpha=0.7, linewidth=0.8, color=color,
                    label=f"{res.intensity_pct:.0f}%")

    ax_raw.axvline(0, color="gray", linestyle="--", linewidth=0.7)
    ax_raw.set_xlim(xlim)
    ax_raw.set_title("RAW overlays", fontsize=10, fontweight="bold")
    ax_raw.set_ylabel("Voltage (µV)")
    ax_raw.legend(fontsize=7, ncol=2, loc="upper right")
    ax_raw.grid(True, alpha=0.2)

    ax_proc.axvline(0, color="gray", linestyle="--", linewidth=0.7)
    ax_proc.set_xlim(mep_xlim)
    ax_proc.set_title("PROCESSED overlays", fontsize=10, fontweight="bold")
    ax_proc.set_xlabel("Time (ms)")
    ax_proc.set_ylabel("Voltage (µV)")
    ax_proc.legend(fontsize=7, ncol=2, loc="upper right")
    ax_proc.grid(True, alpha=0.2)

    # Auto-scale processed panel
    all_seg = []
    for t_ms, raw, proc, res in epochs:
        m = (t_ms >= mep_xlim[0]) & (t_ms <= mep_xlim[1])
        all_seg.append(proc[m] * V_TO_UV)
    if all_seg:
        cat = np.concatenate(all_seg)
        margin = max(20.0, 0.12 * (np.max(cat) - np.min(cat)))
        ax_proc.set_ylim(np.min(cat) - margin, np.max(cat) + margin)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, (ax_raw, ax_proc)


def plot_recruitment_curve(rc: RecruitmentCurve, show: bool = False) -> Tuple[Figure, Tuple[Axes, Axes]]:
    """Plot recruitment curve: amplitude (left) and response probability (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    unique_ints = np.sort(np.unique(rc.intensities))

    # Amplitude panel
    valid = ~rc.trial_rejected
    ax1.scatter(rc.intensities[valid], rc.ptp_uv_processed[valid], s=15, alpha=0.35,
               color="#2c7bb6", label="accepted trials")
    if np.any(rc.trial_rejected):
        ax1.scatter(rc.intensities[rc.trial_rejected], rc.ptp_uv_processed[rc.trial_rejected],
                   s=15, alpha=0.35, color="red", marker="x", label="rejected trials")

    means, sds = [], []
    for ui in unique_ints:
        mask = (rc.intensities == ui) & valid
        vals = rc.ptp_uv_processed[mask]
        means.append(np.mean(vals) if vals.size > 0 else 0)
        sds.append(np.std(vals) if vals.size > 0 else 0)
    ax1.errorbar(unique_ints, means, yerr=sds, fmt="o-", linewidth=1.2, markersize=5,
                color="#2c7bb6", label="mean ± SD")
    ax1.set_xlabel("Intensity (%MSO)")
    ax1.set_ylabel("PTP (µV)")
    ax1.set_title("Recruitment Curve", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.2)

    # Probability panel
    obs_frac = []
    for ui in unique_ints:
        mask = (rc.intensities == ui) & valid
        obs_frac.append(np.mean(rc.response_occurred[mask]) if np.any(mask) else 0)
    ax2.plot(unique_ints, obs_frac, "o-", color="#2c7bb6", label="observed")

    # True sigmoid
    x_fine = np.linspace(unique_ints[0], unique_ints[-1], 200)
    if rc.results:
        slope = rc.results[0].true_slope
        thresh = rc.results[0].true_threshold_pct
        p_true = [_sigmoid(slope * (x - thresh)) for x in x_fine]
        ax2.plot(x_fine, p_true, "--", linewidth=1.0, color="gray", label="true sigmoid")

    ax2.set_xlabel("Intensity (%MSO)")
    ax2.set_ylabel("P(response)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Response Probability", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    if show:
        plt.show()
    return fig, (ax1, ax2)


# ════════════════════════════════════════════════
#  CLI Demo
# ════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    preprocess_cfg = PreprocessConfig(
        enabled=True,
        do_artifact_blanking=True,
        blank_window_ms=(-1.0, 6.0),
        blank_method="linear",
        do_bandpass=True,
        bandpass_hz=(10.0, 500.0),
        bandpass_order=4,
        use_zero_phase=True,
        do_notch=True,
        notch_hz=60.0,
        notch_q=30.0,
        notch_harmonics=(2, 3),
        do_baseline_correction=True,
        baseline_window_ms=(-100.0, -10.0),
        do_trial_rejection=True,
        rejection_threshold_uv=50.0,
    )

    contamination_cfg = ContaminationConfig(
        add_powerline=True,
        powerline_freq_hz=60.0,
        powerline_amplitude_uv=15.0,
        add_drift=True,
        drift_freq_hz=1.0,
        drift_amplitude_uv=40.0,
    )

    # ── Demo 1: Single trial with contamination ────────────────
    print("\n" + "=" * 60)
    print("DEMO 1: Single trial with 60 Hz noise + drift")
    print("=" * 60)

    cfg = EpochConfig(
        intensity_pct=55.0,
        true_threshold_pct=50.0,
        true_slope=0.25,
        noise_rms_uv=12.0,
        latency_ms=22.0,
        artifact_amplitude_V=0.003,
        silent_period=True,
        pre_innervation=False,
        seed=0,
        preprocess=preprocess_cfg,
        contamination=contamination_cfg,
    )

    t_ms, raw_V, proc_V, res = simulate_trial(cfg)
    print(_epoch_title(res))

    fig1, _ = plot_epoch_dual(t_ms, raw_V, proc_V, res)
    fig1.savefig("demo1_epoch_dual.png", dpi=150, bbox_inches="tight")

    # ── Demo 2: Spectrum visualization ─────────────────────────
    print("\n" + "=" * 60)
    print("DEMO 2: Frequency spectrum (raw vs processed)")
    print("=" * 60)

    fig2, _ = plot_spectrum(t_ms, raw_V, proc_V, cfg.sampling_rate)
    fig2.savefig("demo2_spectrum.png", dpi=150, bbox_inches="tight")

    # ── Demo 3: Phase distortion comparison ────────────────────
    print("\n" + "=" * 60)
    print("DEMO 3: Phase distortion — filtfilt vs lfilter")
    print("=" * 60)

    fig3, _ = plot_phase_comparison(t_ms, raw_V, cfg.sampling_rate)
    fig3.savefig("demo3_phase_comparison.png", dpi=150, bbox_inches="tight")

    # ── Demo 4: Envelope extraction ────────────────────────────
    print("\n" + "=" * 60)
    print("DEMO 4: EMG envelope extraction")
    print("=" * 60)

    fig4, _ = plot_envelope(t_ms, proc_V, cfg.sampling_rate, window_ms=15.0,
                           zoom_ms=(-50.0, 80.0))
    fig4.savefig("demo4_envelope.png", dpi=150, bbox_inches="tight")

    # ── Demo 5: Intensity overlay ──────────────────────────────
    print("\n" + "=" * 60)
    print("DEMO 5: Overlay across intensities")
    print("=" * 60)

    epochs = []
    for i, pct in enumerate([40, 45, 50, 52, 55, 60]):
        c = EpochConfig(
            intensity_pct=float(pct),
            true_threshold_pct=50.0,
            true_slope=0.25,
            pre_innervation=(pct == 55),
            seed=100 + i,
            preprocess=preprocess_cfg,
            contamination=contamination_cfg,
        )
        epochs.append(simulate_trial(c))
    fig5, _ = plot_overlay(epochs)
    fig5.savefig("demo5_overlay.png", dpi=150, bbox_inches="tight")

    # ── Demo 6: Recruitment curve ──────────────────────────────
    print("\n" + "=" * 60)
    print("DEMO 6: Recruitment curve")
    print("=" * 60)

    rc = simulate_recruitment_curve(
        intensities=range(30, 75, 3),
        n_trials_per=8,
        true_threshold_pct=50.0,
        true_slope=0.25,
        preprocess=preprocess_cfg,
        contamination=contamination_cfg,
    )
    thr = rc.estimate_threshold(criterion_uv=50.0, use_processed=True)
    print(f"Estimated threshold: {thr}% MSO")
    n_rejected = np.sum(rc.trial_rejected)
    print(f"Trials rejected (pre-stim RMS > 50 µV): {n_rejected}/{len(rc.results)}")

    fig6, _ = plot_recruitment_curve(rc)
    fig6.savefig("demo6_recruitment.png", dpi=150, bbox_inches="tight")

    print("\n✓ All demos complete. Figures saved to current directory.")
    plt.show()