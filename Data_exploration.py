# File to explore data
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupShuffleSplit

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

