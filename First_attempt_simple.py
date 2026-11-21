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


# (3): FEATURE EXTRACTION
def eeg_features(window): #AI-generated code
    """All EEG features for one preprocessed window of shape (samples, 19). 228 features:

    per channel (19 x 10 = 190): log band power (4), relative band power (4), std, line length
    asymmetry (8 pairs x 4 bands = 32): log power left minus right
    global (6): mean over channels of the 4 log band powers, mean std, mean line length
    """
    freqs, psd = welch(window, fs=FS, nperseg=2 * FS, axis=0)             # psd: (frequencies, channels)
    band_power = np.stack([psd[(freqs >= lo) & (freqs < hi)].sum(axis=0) for lo, hi in EEG_BANDS.values()])
    log_power = np.log(band_power + EPS)                                   # (4 bands, 19 channels)
    rel_power = band_power / (band_power.sum(axis=0, keepdims=True) + EPS)
    std = window.std(axis=0)
    line_length = np.abs(np.diff(window, axis=0)).mean(axis=0)             # mean |dx|: independent of window length
    ch = {name: i for i, name in enumerate(EEG_CHANNELS)}

    feats = {}
    for name, j in ch.items():
        for k, band in enumerate(EEG_BANDS):
            feats[f"{name}_{band}_logpow"] = log_power[k, j]
            feats[f"{name}_{band}_relpow"] = rel_power[k, j]
        feats[f"{name}_std"] = std[j]
        feats[f"{name}_linelen"] = line_length[j]
    for left, right in HOMOLOGOUS_PAIRS:
        for k, band in enumerate(EEG_BANDS):
            feats[f"asym_{left}_{right}_{band}"] = log_power[k, ch[left]] - log_power[k, ch[right]]
    for k, band in enumerate(EEG_BANDS):
        feats[f"global_{band}_logpow"] = log_power[k].mean()
    feats["global_std"] = std.mean()
    feats["global_linelen"] = line_length.mean()
    return feats


def eeg_row(eeg_id, path, mid_offset, window_seconds, use_filter): #AI-generated code
    """Preprocess one EEG file and compute its features (a failing file gives NaN features, not a crash)."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    try:
        feats = eeg_features(load_eeg_window(path, mid_offset, window_seconds, use_filter))
    except Exception as err:
        print(f"warning: EEG {eeg_id} failed ({err})")
        feats = {}
    feats["eeg_id"] = eeg_id
    return feats


def compute_eeg_features(meta, eeg_dir, window_seconds, use_filter, is_test, n_jobs): #AI-generated code
    """EEG features for all rows of `meta`, one file per job, in parallel."""
    t0 = time.time()
    mids = np.zeros(len(meta)) if is_test else meta["eeg_mid"].to_numpy()
    jobs = [delayed(eeg_row)(e, eeg_dir / f"{e}.parquet", m, window_seconds, use_filter)
            for e, m in zip(meta["eeg_id"], mids)]
    out = pd.DataFrame(Parallel(n_jobs=n_jobs, batch_size=32)(jobs))
    print(f"EEG features: {len(out)} rows x {out.shape[1] - 1} features in {time.time() - t0:.0f}s")
    return out


def band_columns(columns): #same as hms_sklearn_baseline
    """Return {(region, band): [column indices]} for the 400 spectrogram columns like 'LL_1.17'."""
    out = {}
    for r in REGIONS:
        for b, (lo, hi) in BANDS.items():
            out[(r, b)] = [i for i, c in enumerate(columns)
                           if c.startswith(r + "_") and lo <= float(c.split("_")[1]) < hi]
    return out


def spec_window_features(spec, cols): #cfr hms_sklearn_baseline
    """The 88 baseline features for one 10-minute spectrogram window (time steps x 400)."""
    feats = {}
    total = {r: 0.0 for r in REGIONS}
    band_power = {}
    for (r, b), idx in cols.items():
        p = np.nanmean(spec[:, idx], axis=1)                    # mean power in the band, per time step
        lp = np.log1p(np.nan_to_num(p, nan=0.0).clip(min=0))    # log scale; missing values -> 0
        feats[f"{r}_{b}_mean"] = lp.mean()
        feats[f"{r}_{b}_std"] = lp.std()
        feats[f"{r}_{b}_min"] = lp.min()
        feats[f"{r}_{b}_max"] = lp.max()
        band_power[(r, b)] = np.nanmean(p) if np.isfinite(p).any() else 0.0
        total[r] += band_power[(r, b)]
    for (r, b), v in band_power.items():                        # share of the region's total power
        feats[f"{r}_{b}_rel"] = v / (total[r] + 1e-6)           
    for b in SPEC_BANDS:                                        # left minus right hemisphere
        feats[f"asym_lat_{b}"] = feats[f"LL_{b}_mean"] - feats[f"RL_{b}_mean"]
        feats[f"asym_par_{b}"] = feats[f"LP_{b}_mean"] - feats[f"RP_{b}_mean"]
    return feats


def spectrogram_features(path, windows, is_test): #identical to hms_sklearn_baseline
    """Read one spectrogram file once and compute features for every EEG that uses it."""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    spec = pd.read_parquet(path)
    cols = band_columns(list(spec.columns[1:]))
    spec = spec.iloc[:, 1:].to_numpy(dtype=np.float32)          # drop the 'time' column
    rows = []
    for eeg_id, off_min, off_max in windows:
        if is_test:
            # Test spectrograms are short and centred on the segment: take their middle 10 minutes.
            start = max(0, (len(spec) - 300) // 2)
        else:
            # Training spectrograms cover the whole recording. Start at the midpoint of the labelled
            # offsets; offsets are in seconds and rows are 2 s apart, hence the division by 4 (= /2 /2).
            start = int((off_min + off_max) // 4)
        feats = window_features(spec[start:start + 300], cols)
        feats["eeg_id"] = eeg_id
        rows.append(feats)
    return rows

def compute_spec_features(meta, spec_dir, is_test, n_jobs): #AI-generated code
    """Spectrogram features for all rows of `meta` (columns prefixed with 'spec_')."""
    t0 = time.time()
    if is_test:
        meta = meta.assign(spec_off_min=0.0, spec_off_max=0.0)
    jobs = [delayed(spectrogram_features)(spec_dir / f"{sid}.parquet",
                                          list(zip(g.eeg_id, g.spec_off_min, g.spec_off_max)), is_test)
            for sid, g in meta.groupby("spectrogram_id")]
    results = Parallel(n_jobs=n_jobs, batch_size=32)(jobs)
    out = pd.DataFrame([row for rows in results for row in rows])
    out = out.rename(columns=lambda c: c if c == "eeg_id" else f"spec_{c}")
    print(f"Spectrogram features: {len(jobs)} files -> {len(out)} rows in {time.time() - t0:.0f}s")
    return out


def cached(path, compute): #AI-generated code
    """Load a feature table from the cache if present, otherwise compute (and store) it."""
    if path is not None and path.exists():
        print(f"loaded cached features: {path}")
        return pd.read_parquet(path)
    df = compute()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
    return df


def build_features(meta, data_dir, args, is_test=False): #cfr hms_sklearn_baseline
    """Complete feature table (EEG features, plus spectrogram features unless --no-spec) for `meta`."""
    split = "test" if is_test else "train"
    cache_dir = None if args.no_cache else args.cache_dir
    tag = f"n{len(meta)}_s{args.seed}"
    path = lambda name: None if cache_dir is None else cache_dir / name

    out = cached(path(f"eeg_{split}_{tag}_filter{int(args.filter)}_win{args.window}.parquet"),
                 lambda: compute_eeg_features(meta, data_dir / f"{split}_eegs", args.window,
                                              args.filter, is_test, args.n_jobs))
    if args.spec:
        spec = cached(path(f"spec_{split}_{tag}.parquet"),
                      lambda: compute_spec_features(meta, data_dir / f"{split}_spectrograms",
                                                    is_test, args.n_jobs))
        out = out.merge(spec, on="eeg_id", how="left")
    return out

# (4) THE MODEL (almost completely identical to hms_sklearn_baseline)
def kl_score(y_true, y_pred, eps=1e-15):
    """Mean KL divergence KL(y_true || y_pred) over samples, as in the competition metric."""
    p = np.clip(y_pred, eps, 1)
    p = p / p.sum(axis=1, keepdims=True)
    t = y_true
    return np.mean(np.sum(np.where(t > 0, t * np.log(np.where(t > 0, t, 1) / p), 0), axis=1))


def expand(X, Y, sample_weight=None):
    """Turn soft labels into weighted hard labels.

    Each sample is repeated once per class (label = class index, weight = vote fraction). An optional
    per-sample weight (the vote reliability weight) multiplies the vote fractions. Zero-weight copies are dropped.
    """
    n, k = Y.shape
    Xr = np.repeat(X, k, axis=0)
    yr = np.tile(np.arange(k), n)
    wr = Y.reshape(-1).copy()
    if sample_weight is not None:
        wr = wr * np.repeat(sample_weight, k)
    keep = wr > 0
    return Xr[keep], yr[keep], wr[keep]


def make_model(seed=42):
    """Same hyperparameters as the baseline's gradient boosting model."""
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed)


def fit(X, Y, sample_weight=None, seed=42): #slightly different to hms_sklearn_baseline: sample_weight is now an input to the function
    """Fit on features X and soft labels Y (rows are vote distributions)."""
    Xe, ye, we = expand(X, Y, sample_weight)
    model = make_model(seed)
    model.fit(Xe, ye, sample_weight=we)
    return model


def predict_probs(model, X, n_classes=6): #This one is not in hms_sklearn_baseline, code is AI-generated
    """Probabilities for all six classes (even if one was absent in training), clipped and normalised."""
    p = np.zeros((len(X), n_classes))
    p[:, model.classes_.astype(int)] = model.predict_proba(X)
    p = np.clip(p, 1e-8, None)
    return p / p.sum(axis=1, keepdims=True)


def to_matrix(frame, feats):
    """Feature table -> float32 array with missing / infinite values replaced by 0."""
    return np.nan_to_num(frame[feats].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# (5) CROSS VALIDATION

def cross_validate(X, Y, groups, num_votes, vote_weight, variants, n_folds, seed): #similar to hms_sklearn_baseline, some AI-generated modifications
    """Patient-grouped K-fold CV for each vote-handling variant.

    variants: {name: (weight_by_votes, min_votes)}. The vote handling only affects the TRAINING part of
    each fold; validation always uses all EEGs and the plain (unweighted) KL, so variants stay comparable.
    All variants use exactly the same folds.
    """
    prior = np.tile(Y.mean(axis=0), (len(Y), 1))     # trivial reference: always the mean distribution
    hi = num_votes >= 10                             # EEGs with at least 10 expert votes
    folds = list(GroupKFold(n_splits=n_folds).split(X, Y, groups))
    results = {}
    for name, (weight_by_votes, min_votes) in variants.items():
        t0 = time.time()
        oof = np.zeros_like(Y)                       # out-of-fold predictions
        for fold, (tr, va) in enumerate(folds):
            tr = tr[num_votes[tr] >= min_votes]
            w = vote_weight[tr] if weight_by_votes else None
            oof[va] = predict_probs(fit(X[tr], Y[tr], w, seed), X[va])
            print(f"  [{name}] fold {fold}: KL={kl_score(Y[va], oof[va]):.4f}")
        results[name] = (kl_score(Y, oof), kl_score(Y[hi], oof[hi]))
        print(f"  [{name}] done in {time.time() - t0:.0f}s")

    print(f"\n{'variant':<28}{'KL all':>10}{'KL >=10 votes':>16}")
    print(f"{'prior (mean distribution)':<28}{kl_score(Y, prior):>10.4f}{kl_score(Y[hi], prior[hi]):>16.4f}")
    print(f"{'baseline (earlier run)':<28}{BASELINE_KL_ALL:>10.4f}{BASELINE_KL_HI:>16.4f}")
    for name, (kl_all, kl_hi) in results.items():
        print(f"{name:<28}{kl_all:>10.4f}{kl_hi:>16.4f}")
    return results


# (6): MAIN -- features, cross-validation, final model, submission, AI-generated code
def main():
    parser = argparse.ArgumentParser(description="HMS first own attempt: filtered EEG features + HGB")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="folder with train.csv, test.csv, train_eegs/, test_eegs/, train_spectrograms/, test_spectrograms/")
    parser.add_argument("--mode", choices=["full", "quick"], default="full",
                        help="full: all training EEGs; quick: EEGs of a random 3,000 spectrograms")
    parser.add_argument("--filter", action=argparse.BooleanOptionalAction, default=True,
                        help="bandpass 0.5-40 Hz on the raw EEG (default on)")
    parser.add_argument("--spec", action=argparse.BooleanOptionalAction, default=True,
                        help="add the baseline's spectrogram features to the EEG features (default on)")
    parser.add_argument("--window", type=int, choices=[10, 50], default=10,
                        help="seconds of EEG used for the features (central 10 s, or the whole 50 s segment)")
    parser.add_argument("--weight-votes", action=argparse.BooleanOptionalAction, default=True,
                        help="weight training samples by their number of votes (default on)")
    parser.add_argument("--min-votes", type=int, default=0, help="drop training EEGs with fewer votes (0 = keep all)")
    parser.add_argument("--vote-cap", type=int, default=10, help="number of votes from which a sample gets full weight")
    parser.add_argument("--ablation", action="store_true",
                        help="compare 4 vote-handling variants in CV (final model still uses the flags above)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU cores for reading files (-1 = all)")
    parser.add_argument("--cache-dir", type=Path, default=Path("feature_cache"))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("hms_first_attempt_model.joblib"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # (1) labels and vote information
    df = load_labels(args.data_dir, max_specs=3000 if args.mode == "quick" else None,
                     seed=args.seed, vote_cap=args.vote_cap)
    print(df.shape, "EEGs from", df.patient_id.nunique(), "patients,", df.spectrogram_id.nunique(), "spectrograms")

    # (2) + (3) preprocessing and features
    X_df = build_features(df, args.data_dir, args)
    df = df.merge(X_df, on="eeg_id", how="left")
    feats = [c for c in X_df.columns if c != "eeg_id"]
    print(len(feats), "features")
    X = to_matrix(df, feats)
    Y = df[CLASSES].to_numpy()
    num_votes = df["num_votes"].to_numpy()
    vote_weight = df["vote_weight"].to_numpy()

    # (5) cross-validation (always on)
    if args.ablation:
        variants = {"no vote handling": (False, 0),
                    "weighted by votes": (True, 0),
                    "min 10 votes": (False, 10),
                    "weighted + min 5 votes": (True, 5)}
    else:
        variants = {f"config (weight={args.weight_votes}, min={args.min_votes})":
                    (args.weight_votes, args.min_votes)}
    cross_validate(X, Y, df["patient_id"].to_numpy(), num_votes, vote_weight, variants, args.folds, args.seed)

    # (6) final model on all training data (vote handling as set by the flags)
    keep = num_votes >= args.min_votes
    weights = vote_weight[keep] if args.weight_votes else None
    final = fit(X[keep], Y[keep], weights, args.seed)
    joblib.dump({"model": final, "features": feats, "config": vars(args)}, args.model_out)

    test = pd.read_csv(args.data_dir / "test.csv")
    T_df = build_features(test, args.data_dir, args, is_test=True)
    T_df = test[["eeg_id"]].merge(T_df, on="eeg_id", how="left")        # keep the order of test.csv

    sub = pd.DataFrame({"eeg_id": T_df["eeg_id"]})
    sub[CLASSES] = predict_probs(final, to_matrix(T_df, feats))
    sub.to_csv(args.output, index=False)
    print(f"Saved {args.output} with {len(sub)} rows")


if __name__ == "__main__":   # required for parallel workers on macOS and Windows
    main()