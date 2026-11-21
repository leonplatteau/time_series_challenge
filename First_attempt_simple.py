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
