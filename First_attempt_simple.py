#This is the first attempt to go through the whole process of building a classification model.
#In this attempt, some aspects are already covered, while others are kept behind for a better model.
# Build-up of the code:
# (0) Initialization: loading all packages, loading all files, defining invariant parameters.
# (1) Preprocessing1: defining extra necessary columns in dataframe:
#       This is mainly on amount of votes
# (2) Preprocessing2: filtering the EEG signals (bandpass 0.5-40Hz --> in literature (https://www.researchgate.net/publication/384165067_Advancing_Emotional_Health_Assessments_A_Hybrid_Deep_Learning_Approach_Using_Physiological_Signals_for_Robust_Emotion_Recognition))
# (3) Feature-extraction: in one function, all relevant features are calculated for an input (preprocessed) EEG
# (4) Appplying the actual model: HistGradientBoostingClassifier 
#       --> This is the same as in the hms_sklearn_baseline file, but with a different preprocessing, the result hopefully gets better.
# (5) Cross-validation.
# This is not yet the final attempt, but is a first try into building a model by myself. 
# Maybe it's even worse than the baseline, but hopefully another try can make it better then.


# (0): INITIALIZATION
import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

# Some spectrograms contain all-NaN bands; the resulting empty-mean warnings are handled explicitly below.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Vote columns in train.csv, in the order required by the submission file.
CLASSES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]

# --- raw EEG constants
FS = 200                         # sampling frequency (Hz)
SEGMENT_SECONDS = 50             # length of one annotated EEG segment; the label concerns its central 10 s
EEG_CHANNELS = ["Fp1", "F3", "C3", "P3", "F7", "T3", "T5", "O1", "Fz", "Cz", "Pz",
                "Fp2", "F4", "C4", "P4", "F8", "T4", "T6", "O2"]          # the 19 EEG channels (no EKG)
HOMOLOGOUS_PAIRS = [("Fp1", "Fp2"), ("F3", "F4"), ("C3", "C4"), ("P3", "P4"),
                    ("O1", "O2"), ("F7", "F8"), ("T3", "T4"), ("T5", "T6")]  # (left, right) electrode pairs
EEG_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
BANDPASS_HZ = (0.5, 40.0)        # bandpass applied in step (2)
CLIP_UV = 1000.0                 # crude artefact clipping (microvolt)
EPS = 1e-8

# --- spectrogram constants (same as hms_sklearn_baseline.py)
REGIONS = ["LL", "RL", "LP", "RP"]
SPEC_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 20)}

# --- reference numbers measured earlier with hms_sklearn_baseline.py (CV KL, lower is better)
BASELINE_KL_ALL = 0.9343
BASELINE_KL_HI = 0.9046

# (1): PREPROCESSING 1 -- labels and vote information: based on hms_sklearn_baseline
def load_labels(data_dir, max_specs=None, seed=42, vote_cap=10):
    """Return one row per EEG: ids, label offsets, patient id, vote distribution and vote information.

    Extra columns (the 'amount of votes' part):
      num_votes    total number of expert votes (summed over the labelled sub-segments, as in the baseline)
      vote_weight  min(num_votes, vote_cap) / vote_cap -> reliability weight between 0 and 1
    """
    train = pd.read_csv(data_dir / "train.csv")

    # train.csv has several labelled sub-segments per eeg_id: sum their votes and keep the offset ranges.
    agg = {"spectrogram_id": "first", "patient_id": "first",
           "eeg_label_offset_seconds": ["min", "max"],
           "spectrogram_label_offset_seconds": ["min", "max"],
           "eeg_sub_id": "count"}
    agg.update({c: "sum" for c in CLASSES})
    df = train.groupby("eeg_id").agg(agg)
    df.columns = ["spectrogram_id", "patient_id", "eeg_off_min", "eeg_off_max",
                  "spec_off_min", "spec_off_max", "n_segments"] + CLASSES

    # Votes -> probability distribution (each row sums to 1), plus vote information.
    df["num_votes"] = df[CLASSES].sum(axis=1)
    df[CLASSES] = df[CLASSES].values / df["num_votes"].values[:, None]
    df["vote_weight"] = np.minimum(df["num_votes"], vote_cap) / vote_cap
    df["eeg_mid"] = (df["eeg_off_min"] + df["eeg_off_max"]) / 2   # start (s) of the middle 50 s segment
    df = df.reset_index()

    # Quick mode: keep only the EEGs of a random subset of spectrograms.
    if max_specs:
        keep = pd.Series(df.spectrogram_id.unique()).sample(n=min(max_specs, df.spectrogram_id.nunique()),
                                                             random_state=seed)
        df = df[df.spectrogram_id.isin(keep)].reset_index(drop=True)
    return df


# (2): PREPROCESSING 2 -- cleaning and filtering the raw EEG (this is not in hms_sklearn_baseline.py, AI-generated code)
def clean_channels(x):
    """Fill missing values by linear interpolation (all-NaN channel -> 0) and clip extreme artefacts."""
    out = np.empty_like(x)
    idx = np.arange(len(x))
    for j in range(x.shape[1]):
        col = x[:, j]
        ok = np.isfinite(col)
        out[:, j] = np.interp(idx, idx[ok], col[ok]) if ok.any() else 0.0
    return np.clip(out, -CLIP_UV, CLIP_UV)


def bandpass(x):
    """Zero-phase 4th-order Butterworth bandpass (0.5-40 Hz) along time (axis 0).
    A 60 Hz notch is not needed: it lies far outside the passband."""
    sos = butter(4, BANDPASS_HZ, btype="bandpass", fs=FS, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def load_eeg_window(path, mid_offset, window_seconds, use_filter):
    """Load the 50 s segment starting at mid_offset, clean + filter it, then crop the central window.

    Filtering is done on the full 50 s and only then cropped, to avoid filter edge effects in the window.
    Test files are exactly 50 s long, so mid_offset = 0 there.
    """
    n_seg = SEGMENT_SECONDS * FS
    x = pd.read_parquet(path, columns=EEG_CHANNELS).to_numpy(dtype=np.float64)
    start = int(round(mid_offset * FS))
    start = min(max(start, 0), max(len(x) - n_seg, 0))
    seg = x[start:start + n_seg]
    if len(seg) < n_seg:                                     # unexpectedly short recording
        seg = np.pad(seg, ((0, n_seg - len(seg)), (0, 0)), mode="edge")
    seg = clean_channels(seg)
    if use_filter:
        seg = bandpass(seg)
    w = int(window_seconds * FS)
    c0 = (n_seg - w) // 2
    return seg[c0:c0 + w]

