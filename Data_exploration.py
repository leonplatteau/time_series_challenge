# File to explore data
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, spectrogram,welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt

#Loading file in Python
map_van_script = Path(__file__).parent
df = pd.read_csv(map_van_script / "train.csv")

#Basic data exploration
print(df.head())
print(df.columns)
print("There are " + str(df["eeg_id"].count()) + " EEG samples") #106800
unique_eeg_id = df['eeg_id'].unique()
print("There are " + str(len(unique_eeg_id)) + " unique EEGs (some samples come from the same EEG)") #17089
max_eegs_in_one = max(df['eeg_sub_id']) #742
print("There are maximum " + str(len(df['spectrogram_id'].unique())) + " samples from 1 EEG")

# Adding extra data column: total amount of voters
df["total_votes"] = df["seizure_vote"] + df["lpd_vote"] + df["gpd_vote"] + df["lrda_vote"] + df["grda_vote"] + df["other_vote"]
Seizures = (df["expert_consensus"] == "Seizure").sum()
LPDs = (df["expert_consensus"] == "LPD").sum()
GPDs = (df["expert_consensus"] == "GPD").sum()
LRDAs = (df["expert_consensus"] == "LRDA").sum()
GRDAs = (df["expert_consensus"] == "GRDA").sum()
Other = (df["expert_consensus"] == "Other").sum()
print("There's ", str(Seizures), " seizure votes")
print("There's ", LPDs, "LPD votes")
print("There's ", GPDs, "GPD votes")
print("There's ", LRDAs, "LRDA votes")
print("There's ", GRDAs, "GRDA votes")
print("There's ", Other, "Other votes")

# Looking at idealized/proto/edge cases (k is a number between 0.50 and 1)
# idealized = if more than k of experts pick the same category
#proto = if between (1-k) and k of experts pick "other"
# edge = if votes are around equally distributed between two of the five real categories (not "other") (1-k<x<k)
VOTES = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
NAMED = VOTES[:5]

def add_case_type(df, k):
    df = df.copy()
    total = df[VOTES].sum(axis=1)                      # total number of votes per row
    share = df[VOTES].div(total, axis=0)               # fraction of experts per class
    other = share["other_vote"]
    named_sorted = np.sort(share[NAMED].to_numpy(), axis=1)[:, ::-1]
    top2 = named_sorted[:, 1]                          # second-largest of the five real classes

    idealized = share.max(axis=1) > k                  # any class (incl. other) above k
    proto = ~idealized & (other > 1 - k) & (other < k)
    edge = ~idealized & ~proto & (top2 > 1 - k)

    df["case_type"] = np.select([idealized, proto, edge], ["idealized", "proto", "edge"], default=None)
    return df

df_extended = add_case_type(df, 0.51)
idealized_count = (df_extended["case_type"] == "idealized").sum()
proto_count = (df_extended["case_type"] == "proto").sum()
edge_count = (df_extended["case_type"] == "edge").sum()
total = idealized_count + proto_count + edge_count
idealized_ratio = idealized_count/total
proto_ratio = proto_count/total
edge_ratio = edge_count/total
print(idealized_count, "or", idealized_ratio)
print(proto_count, "or", proto_ratio)
print(edge_count, "or", edge_ratio)

#plotting a spectrogram (coded by Claude)
FS = 200  # samplefrequentie (Hz)
BANDS = [(0.5, 4), (4, 8), (8, 13), (13, 30)]  # delta, theta, alpha, beta


def plot_eeg_overview(eeg_path, offset=0.0, channels=("Fp1", "T3", "O1", "Fp2", "T4", "O2"),
                      spec_channel="T3", save_to=None):
    """Overzichtsplot van één EEG-bestand.

    offset: eeg_label_offset_seconds uit train.csv. De gearceerde zone is het
            gelabelde venster (offset+20 s tot offset+30 s).
    """
    eeg_path = Path(eeg_path)
    df = pd.read_parquet(eeg_path)
    eeg = df.drop(columns="EKG", errors="ignore")      # enkel hersenkanalen
    t = np.arange(len(df)) / FS

    # Bandpass 0,5-40 Hz om drift en 60 Hz-ruis te verwijderen (enkel voor de plot)
    sos = butter(4, [0.5, 40], btype="band", fs=FS, output="sos")
    filt = pd.DataFrame(sosfiltfilt(sos, eeg.to_numpy(float), axis=0), columns=eeg.columns)

    fig, ax = plt.subplots(3, 1, figsize=(11, 10))

    # 1) tijdreeksen, onder elkaar gestapeld
    spacing = 400
    for i, ch in enumerate(channels):
        ax[0].plot(t, filt[ch] - i * spacing, lw=0.5)
    ax[0].set_yticks([-i * spacing for i in range(len(channels))])
    ax[0].set_yticklabels(channels)
    ax[0].axvspan(offset + 20, offset + 30, alpha=0.15)
    ax[0].set_xlabel("s")
    ax[0].set_title(f"Bandpassed (0.5-40 Hz) EEG, {len(df) / FS:.0f} s; "
                    f"gearceerd = gelabeld 10 s-venster (offset={offset:g})")

    # 2) spectrogram van één kanaal
    f_s, t_s, S = spectrogram(filt[spec_channel].to_numpy(), fs=FS, nperseg=400, noverlap=200)
    m = f_s <= 20
    ax[1].pcolormesh(t_s, f_s[m], 10 * np.log10(S[m] + 1e-9), shading="auto")
    ax[1].set_title(f"Spectrogram {spec_channel} (dB)")
    ax[1].set_ylabel("Hz")
    ax[1].set_xlabel("s")

    # 3) gemiddeld vermogensspectrum over alle kanalen (ruwe data, dus 60 Hz-piek zichtbaar)
    f, p = welch(eeg.to_numpy(float), fs=FS, nperseg=512, axis=0)
    pm = p.mean(axis=1)
    ax[2].semilogy(f[f < 60], pm[f < 60])
    for low, high in BANDS:
        ax[2].axvspan(low, high, alpha=0.1)
    ax[2].set_title("Gemiddelde PSD over kanalen (Welch)")
    ax[2].set_xlabel("Hz")

    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=110)
    plt.show()
    return fig
if __name__ == "__main__":
    train = pd.read_csv("train.csv")
    rij = train[train.eeg_id == 568657].iloc[0]
    plot_eeg_overview(r"C:\Users\leonp\Documents\KUL\3.1\AI\train_eegs\568657.parquet", offset=rij.eeg_label_offset_seconds)