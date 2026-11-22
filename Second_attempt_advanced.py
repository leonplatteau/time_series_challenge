#This is the second attempt with the aim of building a more advanced classification model, based on a 2D-CNN 
# In this attempt, multiple aspects are implemented, trying to make a complete as possible model with the lowest KL-div
# Build-up of the code:
# (0) Initialization: loading all packages, loading all files, defining invariant parameters.
# (1) Preprocessing1: defining extra necessary columns in dataframe:
#       This is mainly on labels and amount of votes
# (2) Preprocessing2: filtering the EEG signals (bandpass 0.5-40Hz --> in literature (https://www.researchgate.net/publication/384165067_Advancing_Emotional_Health_Assessments_A_Hybrid_Deep_Learning_Approach_Using_Physiological_Signals_for_Robust_Emotion_Recognition))
# (3) Representation
# (4) Appplying the actual CNN-model
# (5) Training: loop with stage 1 and stage 2, learning-rate, checkpoints, mixed precision
# (6) Cross-validation.
# (7) Inference and submission
# This is possibly the final attempt, therefore I try to make it as complete as possible. 

# ======================================================================================
# Design decisions are marked with "DECISION".
# ======================================================================================

# (0): INITIALIZATION
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
import timm   # pretrained backbones. Kaggle: weights need internet (training notebook) or a weights dataset

import First_attempt_simple as fa   # reuse: load_labels, kl_score, clean_channels, CLASSES, EEG_CHANNELS
                                    # (same labels + same folds => directly comparable with the HGB results)
import os
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.signal import butter, sosfiltfilt, spectrogram
CLASSES = fa.CLASSES
FS = 200                            # EEG sampling frequency (Hz)

# Double banana (longitudinal bipolar montage): 4 chains x 4 (anode, cathode) pairs = 16 channels.
BIPOLAR_CHAINS = {
    "LL": [("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1")],
    "LP": [("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1")],
    "RP": [("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2")],
    "RL": [("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2")],
}

# Reference numbers from the first attempt (CV KL all / >=10 votes), printed next to the CNN result.
HGB_KL_ALL, HGB_KL_HI = 0.8612, 0.8128

#Relevant constants for PREPROCESSING 2, mainly for STFT
CHAIN_ORDER = list(BIPOLAR_CHAINS)            # ["LL", "LP", "RP", "RL"]: the order of the image channels
OFFICIAL_ROWS = 300                           # 10 minutes of the official spectrogram (rows are 2 s apart)
OFFICIAL_FREQS = 100                          # 0.59 ... 19.92 Hz per chain
STFT_NPERSEG = 128                            # own spectrograms: window (samples) ...
STFT_HOP_FRAC = 0.3                           # ... hop as a fraction of the window ...
STFT_NFFT = 1024                              # ... zero-padded FFT: bin width 200/1024 Hz, same grid as the official ones
FREQ_BINS = slice(3, 103)                     # FFT bins 3..102 = 0.59 ... 19.92 Hz: the same 100 frequencies as the official
LOG_MIN, LOG_MAX = -4.0, 8.0                  # power is clipped to [e^-4, e^8] before the log: keeps peaks, removes extremes

def get_args(): # This function is partially copied from the first_attempt_file, the other part, i wrote by myself. The fully AI-generated lines are annotated.
    """Command-line switches (same philosophy as the first attempt: ablations without code changes).

    --data-dir, --cache-dir, --output, --seed, --n-jobs
    --representation {official, eeg}   official = Kaggle spectrograms, eeg = own spectrograms from the EEG
    --montage / --no-montage           double banana on/off (only for --representation eeg)
    --band-low, --band-high            bandpass edges (0.5 / 20 or 40)
    --backbone (default efficientnet_b0), --pretrained / --no-pretrained, --weights-path
    --image-layout {channels, stacked} chains as input channels, or chains stacked vertically into one image
    --epochs1, --epochs2, --lr1, --lr2, --batch-size, --stage2-min-votes (default 10), --vote-cap
    --folds, --only-fold K (run one fold), --smoke (tiny run to test the pipeline), --amp / --no-amp
    """
    parser = argparse.ArgumentParser(description="HMS second attempt")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="folder with train.csv, test.csv, train_eegs/, test_eegs/, train_spectrograms/, test_spectrograms/")
    parser.add_argument("--cache-dir", type=Path, default=Path("image_cache"),
                        help="cached image arrays, out-of-fold predictions and fold checkpoints")    
    parser.add_argument("--image-layout", choices=["channels", "stacked"], default="channels",
                            help="chains as input channels, or chains stacked vertically into one image")
    parser.add_argument("--montage", action=argparse.BooleanOptionalAction, default=True,
                            help="Double banana on/off (default on)")
    parser.add_argument("--band-low", type=float, default=0.5,
                            help=" Lower boundary for bandpassfilter")
    parser.add_argument("--band-high", type=float, default=40,
                                help=" Upper boundary for bandpassfilter")
    parser.add_argument("--weights-path", type=Path, default=None,
                        help="optional local file with pretrained weights (for notebooks without internet)")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0",
                        help="name of a timm model, e.g. efficientnet_b0, efficientnet_b2, resnet34") #AI-generated line
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage2-min-votes", type=int, default=10, help="stage 2 fine-tunes only on EEGs with at least this many votes")
    parser.add_argument("--vote-cap", type=int, default=10, help="number of votes from which a sample gets full weight")
    parser.add_argument("--cv-only", action="store_true",
                        help="stop after cross-validation (no test predictions, no submission)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--only-fold", type=int, default=None, help= "run one fold")
    parser.add_argument("--smoke", action=argparse.BooleanOptionalAction, default = False, help="tiny run to test the pipeline")
    parser.add_argument("--representation", choices=["official", "eeg"], default = "official", help = "official = Kaggle spectrograms, eeg = own spectrograms from the EEG" )
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU cores for reading files (-1 = all)")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader worker processes (use 0 if Windows gives trouble)") #AI-generated line
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--epochs1", type=int, default=10)
    parser.add_argument("--epochs2", type=int, default=3)
    parser.add_argument("--lr1", type=float, default=1e-3)
    parser.add_argument("--lr2", type=float, default=1e-4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default = True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.smoke:                      # one fold, one epoch per stage, AI-generated if-statement
        args.epochs1 = args.epochs2 = 1
        if args.only_fold is None:
            args.only_fold = 0
    return args

def set_seed(seed):
    """ Seed python, numpy and torch (incl. cuda) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)      # does nothing (and does not fail) without a GPU


def get_device():
    """cuda if available, else cpu (print which one)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"Using device: {device} ({name})")
    return device

# (1): PREPROCESSING 1 -- labels, vote information, folds #AI-generated functions
def load_data(args): 
    """One row per EEG with its vote distribution, vote information and offsets (reuses the first attempt).

    Columns (from fa.load_labels): eeg_id, spectrogram_id, patient_id, eeg_off_min/max, spec_off_min/max,
    n_segments, the 6 vote fractions (CLASSES), num_votes, vote_weight, eeg_mid.
    --smoke: keep a random subset of 2000 EEGs, enough to test the whole pipeline quickly.
    """
    df = fa.load_labels(args.data_dir, vote_cap=args.vote_cap, seed=args.seed)
    if args.smoke:
        df = df.sample(n=min(2000, len(df)), random_state=args.seed).reset_index(drop=True)
    print(df.shape, "EEGs from", df.patient_id.nunique(), "patients,", df.spectrogram_id.nunique(), "spectrograms")
    return df


def make_folds(df, n_folds):
    """List of (train_idx, val_idx) from GroupKFold on patient_id.

    Same splitter as in the first attempt: in full mode the folds are identical, so CNN and HGB can be compared
    per fold. A patient never appears in both the training and the validation part of a fold.
    """
    folds = list(GroupKFold(n_splits=n_folds).split(df, groups=df["patient_id"].to_numpy()))
    for k, (tr, va) in enumerate(folds):
        overlap = set(df["patient_id"].iloc[tr]) & set(df["patient_id"].iloc[va])
        assert not overlap, f"fold {k}: patients in both train and validation: {sorted(overlap)[:5]}"
        print(f"fold {k}: {len(tr)} train / {len(va)} validation EEGs")
    return folds


# (2): PREPROCESSING 2 -- from raw files to model-ready images (computed ONCE, then cached)
def _read_spectrogram_file(path):
    """Official spectrogram parquet -> (array (time, 400), list of column names), without the 'time' column."""
    spec = pd.read_parquet(path)
    return spec.iloc[:, 1:].to_numpy(dtype=np.float32), list(spec.columns[1:])


def _official_window(arr, cols, offset_s, is_test):
    """Cut the 10-minute window out of a spectrogram array and turn it into a (4, 100, 300) log image."""
    if is_test:
        start = max(0, (len(arr) - OFFICIAL_ROWS) // 2)          # test spectrograms: take the middle 10 minutes
    else:
        start = int(offset_s // 2)                                # offset in seconds, rows are 2 s apart
    start = min(max(start, 0), max(len(arr) - OFFICIAL_ROWS, 0))
    win = arr[start:start + OFFICIAL_ROWS]
    if len(win) < OFFICIAL_ROWS:                                  # unexpectedly short file: pad with NaN
        win = np.pad(win, ((0, OFFICIAL_ROWS - len(win)), (0, 0)), constant_values=np.nan)
    chains = []
    for chain in CHAIN_ORDER:
        idx = sorted((i for i, c in enumerate(cols) if c.startswith(chain + "_")),
                     key=lambda i: float(cols[i].split("_")[1]))   # by frequency
        chains.append(win[:, idx].T)                               # (frequency, time)
    img = np.nan_to_num(np.stack(chains), nan=0.0)                 # NaN -> 0 power -> floor value after clipping
    return np.log(np.clip(img, np.exp(LOG_MIN), np.exp(LOG_MAX))).astype(np.float32)


def load_official_spectrogram(path, offset_s, is_test):
    """Official Kaggle spectrogram -> array (4 chains, 100 frequencies, 300 time steps), chain order LL, LP, RP, RL.

    offset_s: train = midpoint of spectrogram_label_offset_seconds; ignored for test.
    Power is clipped to [e^-4, e^8] and log-transformed (keeps the peaks, removes extremes); NaN -> floor value.
    """
    arr, cols = _read_spectrogram_file(path)
    return _official_window(arr, cols, offset_s, is_test)


def load_eeg_50s(path, mid_offset):
    """50 s EEG segment (10000 x 19, channel order fa.EEG_CHANNELS), NaN interpolated and clipped.
    mid_offset: start (s) of the segment in the file (train: eeg_mid; test: 0)."""
    return fa.load_eeg_window(path, mid_offset, window_seconds=fa.SEGMENT_SECONDS, use_filter=False)


def filter_eeg(x, low, high):
    """Zero-phase 4th-order Butterworth bandpass along time (axis 0) of x (time, channels); edges in Hz."""
    assert 0 < low < high < fa.FS / 2, f"invalid bandpass edges: {low}-{high} Hz"
    sos = butter(4, [low, high], btype="bandpass", fs=fa.FS, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def bipolar_montage(x, channel_names):
    """Double banana: x (time, 19) -> (16, time) as anode minus cathode, order = chains LL, LP, RP, RL (4 pairs each)."""
    idx = {name: i for i, name in enumerate(channel_names)}
    pairs = [pair for chain in CHAIN_ORDER for pair in BIPOLAR_CHAINS[chain]]
    return np.stack([x[:, idx[a]] - x[:, idx[b]] for a, b in pairs])


def referential_chains(x, channel_names):
    """Control for --no-montage: the same chains, but the plain electrodes (no subtraction).
    x (time, 19) -> (20, time): 5 electrodes per chain (the 4 pairs share electrodes), order LL, LP, RP, RL."""
    idx = {name: i for i, name in enumerate(channel_names)}
    electrodes = []
    for chain in CHAIN_ORDER:
        pairs = BIPOLAR_CHAINS[chain]
        electrodes += [pairs[0][0]] + [cathode for _, cathode in pairs]
    return np.stack([x[:, idx[e]] for e in electrodes])


def eeg_to_spectrogram(signals):
    """Own spectrograms: signals (n, time) with n divisible by 4 (consecutive blocks = the 4 chains)
    -> (4 chains, 100 frequencies, time steps), same log transform and frequency grid as the official ones.

    STFT with a 128-sample window, hop of 30% and FFT length 1024 (a literature setting; tune later), 0.59-19.92 Hz.
    Power is averaged over the channels of each chain BEFORE the log, as in the official spectrograms.
    """
    hop = int(STFT_NPERSEG * STFT_HOP_FRAC)
    _, _, sxx = spectrogram(signals, fs=fa.FS, window="hann", nperseg=STFT_NPERSEG,
                            noverlap=STFT_NPERSEG - hop, nfft=STFT_NFFT, scaling="density", mode="psd", axis=-1)
    sxx = sxx[:, FREQ_BINS, :]                                              # (n, 100, time steps)
    per_chain = sxx.shape[0] // len(CHAIN_ORDER)
    sxx = sxx.reshape(len(CHAIN_ORDER), per_chain, *sxx.shape[1:]).mean(axis=1)
    return np.log(np.clip(sxx, np.exp(LOG_MIN), np.exp(LOG_MAX))).astype(np.float32)


def _official_job(path, items, is_test):
    """One spectrogram file, all EEGs that use it. items = [(row, offset_s)]. Returns [(row, image or None)]."""
    try:
        arr, cols = _read_spectrogram_file(path)
        return [(row, _official_window(arr, cols, off, is_test).astype(np.float16)) for row, off in items]
    except Exception as err:
        print(f"warning: spectrogram {Path(path).name} failed ({type(err).__name__}: {err})")
        return [(row, None) for row, _ in items]


def _eeg_job(row, path, mid_offset, montage, band_low, band_high):
    """One EEG file -> [(row, image or None)]: load, bandpass, (double banana), STFT."""
    try:
        x = filter_eeg(load_eeg_50s(path, mid_offset), band_low, band_high)
        signals = bipolar_montage(x, fa.EEG_CHANNELS) if montage else referential_chains(x, fa.EEG_CHANNELS)
        return [(row, eeg_to_spectrogram(signals).astype(np.float16))]
    except Exception as err:
        print(f"warning: EEG {Path(path).name} failed ({type(err).__name__}: {err})")
        return [(row, None)]


def build_image_cache(df, data_dir, args, is_test=False):
    """Compute the image of every EEG once and store all of them in ONE float16 array (N, 4, 100, time).

    Returns (images, index) with index = {eeg_id: row in images}. The array is written to
    cache_dir/images_<split>_<representation>_n<N>_s<seed>.npy (via a temporary file, so an interrupted run never
    leaves a broken cache); with --no-cache it only lives in memory. Failed files stay all-zero and are counted.
    The cache does not depend on --image-layout: stacking the chains is done later, in the Dataset.
    """
    t0 = time.time()
    split = "test" if is_test else "train"
    if args.representation == "official":
        tag = "official"
        img_shape = (len(CHAIN_ORDER), OFFICIAL_FREQS, OFFICIAL_ROWS)
    else:
        tag = f"eeg_montage{int(args.montage)}_band{args.band_low:g}-{args.band_high:g}"
        probe = eeg_to_spectrogram(np.zeros((16, fa.SEGMENT_SECONDS * fa.FS)))      # only to learn the shape
        img_shape = probe.shape
    index = {int(e): row for row, e in enumerate(df["eeg_id"])}
    use_cache = not args.no_cache
    cache_path = args.cache_dir / f"images_{split}_{tag}_n{len(df)}_s{args.seed}.npy"

    if use_cache and cache_path.exists():
        print(f"loaded cached images: {cache_path}")
        return np.load(cache_path, mmap_mode="r"), index

    shape = (len(df),) + img_shape
    if use_cache:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(cache_path.name + ".tmp")
        out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16, shape=shape)   # starts as zeros
    else:
        out = np.zeros(shape, dtype=np.float16)

    rows = np.arange(len(df))
    if args.representation == "official":
        offsets = np.zeros(len(df)) if is_test else ((df["spec_off_min"] + df["spec_off_max"]) / 2).to_numpy()
        jobs = [delayed(_official_job)(data_dir / f"{split}_spectrograms" / f"{sid}.parquet",
                                       list(zip(rows[mask], offsets[mask])), is_test)
                for sid, mask in ((s, (df["spectrogram_id"] == s).to_numpy()) for s in df["spectrogram_id"].unique())]
    else:
        mids = np.zeros(len(df)) if is_test else df["eeg_mid"].to_numpy()
        jobs = [delayed(_eeg_job)(row, data_dir / f"{split}_eegs" / f"{eeg_id}.parquet", mids[row],
                                  args.montage, args.band_low, args.band_high)
                for row, eeg_id in zip(rows, df["eeg_id"])]

    n_fail = 0
    for results in Parallel(n_jobs=args.n_jobs, batch_size=32, return_as="generator")(jobs):   # results arrive in order
        for row, img in results:
            if img is None:
                n_fail += 1
            else:
                out[row] = img
    if n_fail == len(df):
        raise RuntimeError(f"No image could be computed ({tag}); check the folders and file names")
    if n_fail:
        print(f"warning: {n_fail} of {len(df)} images failed and are all-zero")

    if use_cache:
        out.flush()
        del out
        os.replace(tmp_path, cache_path)
        out = np.load(cache_path, mmap_mode="r")
    print(f"Images ({tag}, {split}): {shape} in {time.time() - t0:.0f}s")
    return out, index


# (3): REPRESENTATION -- dataset, augmentation, loaders (AI-generated)
SHIFT_FRAC = 0.05            # time-shift augmentation: up to 5% of the image width
FLIP_PROB = 0.5              # probability of mirroring left <-> right


def augment_flip_lr(img):
    """Mirror left <-> right: img (4 chains, freq, time) in chain order LL, LP, RP, RL.

    Reversing the chain axis gives RL, RP, LP, LL, i.e. LL<->RL and LP<->RP. The labels are not side-specific
    (a left-sided LPD mirrored is a right-sided LPD, still an LPD), so the target stays the same.
    """
    return np.ascontiguousarray(img[::-1])


def augment_time_shift(img, max_shift):
    """Random shift along the time axis by -max_shift ... +max_shift steps; the gap is filled with the edge column
    (np.roll would wrap the end of the recording around to the start, which creates a false discontinuity)."""
    if max_shift <= 0:
        return img
    shift = np.random.randint(-max_shift, max_shift + 1)
    if shift == 0:
        return img
    padded = np.pad(img, ((0, 0), (0, 0), (max_shift, max_shift)), mode="edge")
    start = max_shift + shift
    return padded[:, :, start:start + img.shape[2]]


def normalize_image(img):
    """Fixed scaling of the log-power images from [LOG_MIN, LOG_MAX] to [-1, 1] (same for train and test, no statistics
    are learned from the data). DECISION to revisit later: per-image standardisation."""
    return (img - LOG_MIN) / (LOG_MAX - LOG_MIN) * 2.0 - 1.0


def image_to_layout(img, layout):
    """channels: keep (4, freq, time). stacked: put the chains below each other -> (1, 4 * freq, time)."""
    if layout == "channels":
        return img
    return img.reshape(1, -1, img.shape[2])


def n_input_channels(args):
    """Number of input channels of the network: 4 (one per chain) or 1 (chains stacked into one image)."""
    return len(CHAIN_ORDER) if args.image_layout == "channels" else 1


class HMSDataset(Dataset):
    """Returns (image tensor, target distribution (6), sample weight) for the chosen rows of df.

    images, index: output of build_image_cache. rows: positions in df (a fold, or a stage-2 subset).
    train: augment (flip + time shift) when args.augment is on. use_weights: weight = vote_weight (stage 1), else 1.
    A df without vote columns (test.csv) gives all-zero targets.
    """

    def __init__(self, images, index, df, rows, train, args, use_weights=False):
        self.images = images
        self.rows = np.asarray(rows)
        self.img_rows = np.array([index[int(e)] for e in df["eeg_id"].to_numpy()[self.rows]])
        has_targets = all(c in df.columns for c in CLASSES)
        self.targets = (df[CLASSES].to_numpy(dtype=np.float32)[self.rows] if has_targets
                        else np.zeros((len(self.rows), len(CLASSES)), dtype=np.float32))
        self.weights = (df["vote_weight"].to_numpy(dtype=np.float32)[self.rows] if use_weights
                        else np.ones(len(self.rows), dtype=np.float32))
        self.augment = bool(train and args.augment)
        self.layout = args.image_layout
        self.max_shift = int(SHIFT_FRAC * images.shape[-1])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        img = np.asarray(self.images[self.img_rows[i]], dtype=np.float32)       # float16 on disk -> float32
        if self.augment:
            if np.random.rand() < FLIP_PROB:
                img = augment_flip_lr(img)
            img = augment_time_shift(img, self.max_shift)
        img = image_to_layout(normalize_image(img), self.layout)
        return (torch.from_numpy(np.ascontiguousarray(img)),
                torch.from_numpy(self.targets[i]),
                torch.tensor(self.weights[i]))

    def __getstate__(self):
        """Used when DataLoader workers are started with 'spawn' (Windows): pickling a memory-mapped cache would copy
        the whole multi-GB array into every worker. Keep only the file name and re-open it in the worker."""
        state = self.__dict__.copy()
        if isinstance(self.images, np.memmap):
            state["_images_path"] = self.images.filename
            state["_images_shape"] = self.images.shape
            state["images"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.images is None:
            self.images = np.load(self._images_path, mmap_mode="r")


def make_loader(dataset, args, shuffle):
    """DataLoader for training (shuffle=True: shuffled, last incomplete batch dropped) or evaluation (in order).
    A seeded generator makes the shuffle order and the worker seeds reproducible."""
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(args.seed)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle, generator=generator,
                      persistent_workers=args.num_workers > 0)



# (4): THE MODEL
MODEL_INPUT_SIZE = (256, 512)   # (height, width) the images are resized to before the backbone; None = no resizing


class HMSNet(nn.Module):
    """timm backbone (EfficientNet by default) + a new 6-class head. Input: (batch, in_chans, H, W) images from
    HMSDataset; output: logits (batch, 6) (softmax is applied in the loss / when predicting).

    backbone: name of a timm model. in_chans: 4 (channels layout) or 1 (stacked layout); timm adapts the first
    convolution of the pretrained network to this number of channels.
    pretrained: start from ImageNet weights. Without internet give weights_path (a weights file of that backbone,
    e.g. downloaded from the timm page on huggingface.co and uploaded as a Kaggle dataset); the classifier head
    is replaced by a new 6-class head anyway.
    The images are small (100 pixels high), so they are upscaled to MODEL_INPUT_SIZE inside the model.
    """

    def __init__(self, backbone, in_chans, pretrained, weights_path=None, input_size=MODEL_INPUT_SIZE):
        super().__init__()
        kwargs = dict(in_chans=in_chans, num_classes=len(CLASSES))
        if pretrained and weights_path is not None:
            kwargs["pretrained_cfg_overlay"] = dict(file=str(weights_path))      # load the weights from a local file
        self.net = timm.create_model(backbone, pretrained=pretrained, **kwargs)
        self.input_size = input_size

    def forward(self, x):
        if self.input_size is not None and tuple(x.shape[-2:]) != tuple(self.input_size):
            x = F.interpolate(x, size=self.input_size, mode="bilinear", align_corners=False)
        return self.net(x)


def kl_loss(logits, target, weight=None):
    """KL(target || softmax(logits)) per sample, then the mean over the batch; with weights, the weighted mean.

    Same quantity as the competition metric (fa.kl_score). Classes without votes (target 0) contribute nothing.
    weight: per-sample weights (e.g. vote_weight in stage 1); None = plain mean (stage 2).
    """
    log_p = F.log_softmax(logits.float(), dim=1)
    t = target.float()
    log_t = torch.log(t.clamp_min(1e-15))                    # only avoids log(0); those terms are multiplied by 0
    kl = (t * (log_t - log_p)).sum(dim=1)                    # (batch,)
    if weight is None:
        return kl.mean()
    w = weight.float()
    return (kl * w).sum() / w.sum()



# (5): TRAINING -- stage 1 (all data, vote weights) and stage 2 (fine-tune on reliable EEGs)
def run_tag(args):
    """Short name of this experiment (representation, backbone, layout), used in checkpoint and OOF file names so
    different runs never overwrite each other. Smoke runs get their own suffix."""
    if args.representation == "official":
        rep = "official"
    else:
        rep = f"eeg_montage{int(args.montage)}_band{args.band_low:g}-{args.band_high:g}"
    return f"{rep}_{args.backbone}_{args.image_layout}" + ("_smoke" if args.smoke else "")


def _flip_batch(x):
    """Mirror left <-> right for a batch (b, c, h, w): reverse the chains. channels layout: c = 4, flip the channel axis;
    stacked layout: c = 1 and the 4 chains lie below each other in the height axis."""
    if x.shape[1] == len(CHAIN_ORDER):
        return x.flip(1)
    b, c, h, w = x.shape
    return x.reshape(b, c, len(CHAIN_ORDER), h // len(CHAIN_ORDER), w).flip(2).reshape(b, c, h, w)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, use_weights, amp):
    """One pass over the training loader: forward (mixed precision on GPU), weighted KL loss, backward with gradient
    clipping, optimizer step, learning-rate step (the scheduler is stepped per batch). Returns the mean training loss."""
    model.train()
    use_amp = amp and device.type == "cuda"
    total, count = 0.0, 0
    for x, y, w in loader:
        x, y, w = x.to(device, non_blocking=True), y.to(device, non_blocking=True), w.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = kl_loss(logits, y, w if use_weights else None)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)                                   # so the clipping works on the real gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(x)
        count += len(x)
    return total / max(count, 1)


@torch.no_grad()
def predict_loader(model, loader, device, amp, tta_flip=False):
    """Softmax probabilities (n, 6) in loader order (use a loader with shuffle=False).
    tta_flip: average the prediction for the image and for its left-right mirror image."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    out = []
    for x, _, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            logits_flipped = model(_flip_batch(x)) if tta_flip else None
        p = torch.softmax(logits.float(), dim=1)
        if tta_flip:
            p = (p + torch.softmax(logits_flipped.float(), dim=1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out)


def run_stage(model, train_loader, val_loader, epochs, lr, use_weights, device, args, name="stage"):
    """One training stage: AdamW + one-cycle schedule (short warm-up, then cosine decay to ~0), a FIXED number of epochs.
    After every epoch the training loss and the validation KL are printed. The validation fold is only used for
    reporting: the weights of the last epoch are kept, so the CV score is not tuned on the validation fold.
    Returns a list with (train loss, validation KL) per epoch."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = max(1, epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp and device.type == "cuda")
    history = []
    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, use_weights, args.amp)
        val_kl = float("nan")
        if val_loader is not None:
            val_kl = fa.kl_score(val_loader.dataset.targets, predict_loader(model, val_loader, device, args.amp))
        history.append((train_loss, val_kl))
        print(f"  [{name}] epoch {epoch + 1}/{epochs}  train loss {train_loss:.4f}  val KL {val_kl:.4f}"
              f"  ({time.time() - t0:.0f}s)")
    return history


def fit_fold(fold, tr_idx, va_idx, images, index, df, args, device):
    """Train one fold: new model -> stage 1 (all training EEGs, vote weights) -> stage 2 (only EEGs with at least
    args.stage2_min_votes votes, no weights, lower learning rate) -> predict the validation fold -> save the weights.
    Returns the validation probabilities (len(va_idx), 6) in the order of va_idx."""
    t0 = time.time()
    set_seed(args.seed + fold)                                     # --only-fold K reproduces fold K of a full run
    model = HMSNet(args.backbone, n_input_channels(args), args.pretrained, args.weights_path).to(device)

    val_loader = make_loader(HMSDataset(images, index, df, va_idx, False, args), args, shuffle=False)

    train1 = HMSDataset(images, index, df, tr_idx, True, args, use_weights=True)
    run_stage(model, make_loader(train1, args, shuffle=True), val_loader, args.epochs1, args.lr1, True,
              device, args, name=f"fold {fold} stage 1")

    reliable = tr_idx[df["num_votes"].to_numpy()[tr_idx] >= args.stage2_min_votes]
    if args.epochs2 > 0 and len(reliable) > 0:
        print(f"  stage 2 uses {len(reliable)} of {len(tr_idx)} training EEGs (>= {args.stage2_min_votes} votes)")
        train2 = HMSDataset(images, index, df, reliable, True, args, use_weights=False)
        run_stage(model, make_loader(train2, args, shuffle=True), val_loader, args.epochs2, args.lr2, False,
                  device, args, name=f"fold {fold} stage 2")

    probs = predict_loader(model, val_loader, device, args.amp, tta_flip=args.augment)   # flip-TTA only if trained with flips
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.cache_dir / f"model_{run_tag(args)}_fold{fold}.pt")
    print(f"fold {fold} done in {time.time() - t0:.0f}s, validation KL {fa.kl_score(df[CLASSES].to_numpy()[va_idx], probs):.4f}")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs


# (6): CROSS-VALIDATION
def cross_validate_cnn(df, images, index, args, device):
    """[medium] Loop over the folds (or --only-fold), collect out-of-fold probabilities, save them to
    cache_dir/oof_{tag}.npy, print per-fold KL and the table:
        prior | HGB (first attempt) | CNN        for KL all and KL >=10 votes (fa.kl_score)
    Returns the OOF array."""
    raise NotImplementedError


def blend_oof(oof_cnn, oof_hgb, y, hi_mask):
    """[easy] Find the blend weight w in [0, 1] for  w*cnn + (1-w)*hgb  that minimises KL on the OOF predictions.
    NOTE: the first attempt does not save its OOF predictions yet (small addition needed there)."""
    raise NotImplementedError


# (7): INFERENCE AND SUBMISSION
def predict_test(test, images_test, index_test, args, device):
    """[medium] Load every fold checkpoint (pretrained=False, then load_state_dict), predict the test images
    (optional flip-TTA), average the folds' probabilities, clip + renormalise. Returns (n_test, 6)."""
    raise NotImplementedError


def write_submission(test, probs, path):
    """[easy] DataFrame(eeg_id + 6 class columns), rows in the order of test.csv, each row sums to 1."""
    raise NotImplementedError


def main():
    """[easy] args -> seed/device -> load_data -> folds -> build_image_cache (train)
    -> cross_validate_cnn (always on) -> build_image_cache (test) -> predict_test -> write_submission."""
    raise NotImplementedError


if __name__ == "__main__":
    main()