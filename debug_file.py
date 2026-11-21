import First_attempt_simple as fa
w = fa.load_eeg_window(r"C:/Users/leonp/time_series_challenge/test_eegs/3911565283.parquet", 0, 10, True)
print(w.shape)

import argparse
from pathlib import Path
import pandas as pd
import First_attempt_simple as fa

if __name__ == "__main__":
    args = argparse.Namespace(no_cache=True, cache_dir=Path("feature_cache"), seed=42,
                              window=10, filter=True, spec=True, n_jobs=-1)
    test = pd.read_csv("test.csv")
    out = fa.build_features(test, Path("."), args, is_test=True)
    print(out.shape)