#!/usr/bin/env python
"""
Compute metrics for every  speaker_2m??_*e1_emb_100epo  folder and save to metrics.csv
Adds val_loss_min and shortens folder names to   12_dfm_e1   /   12_e1   style.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import umap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

from PR_Dim import PR_Dim
    
def run_metrics(emb_path: Path, loc_path: Path):
    X = np.load(emb_path)
    pr, pr_corrected, d_expvar = PR_Dim(X)
    X_labels = np.squeeze(np.load(loc_path))
    r2_raw = round(r2_score(X_labels, LinearRegression().fit(X, X_labels).predict(X)), 4)
    return dict(PR = pr, PR_C = pr_corrected, D_exp = d_expvar, r2_raw = r2_raw)


def shortest_name(full_folder_name: str) -> str:
    """
    'speaker_2m12_dfm_e1_emb_100epo' -> '12_dfm_e1'
    'speaker_2m23_e1_emb_100epo'     -> '23_e1'
    """
    parts = full_folder_name.split("_")
    idx_e1 = parts.index("e1")  # position of 'e1'
    tokens = parts[1 : idx_e1 + 1]  # ['2m12', 'dfm', 'e1'] or ['2m23', 'e1']
    tokens[0] = tokens[0][2:]  # drop '2m' prefix --> '12' / '23'
    return "_".join(tokens)


def load_val_loss_min(folder: Path) -> float:
    """
    Read the first file starting with 'val_loss' and return its minimum value.
    Each line is assumed to be of the form “[number]”.
    """
    file = next(folder.glob("val_loss*"))   # raises StopIteration if none found
    with open(file) as fh:
        vals = [float(line.strip()[1:-1])     # strip '[' and ']'
                for line in fh if line.strip()]
    if not vals:
        raise ValueError(f"{file.name} is empty or not in expected format")
    return round(min(vals), 4)



# ───────────────────────────────────────────────────────────────────  main
root = Path.cwd()
rows = []

for fld in sorted(root.glob("speaker_2m*_e1_emb_100epo")):
    emb = fld / "speaker-4m-e1_best_epoch_emb10.npy"
    loc = fld / "speaker-4m-e1_best_epoch_emb_locations.npy"
    if not (emb.exists() and loc.exists()):
        print(f"⚠️  Missing .npy files in {fld.name} – skipped.")
        continue

    try:
        row = run_metrics(emb, loc)
        row["val_loss_min"] = round(load_val_loss_min(fld), 4)
        row["folder"] = shortest_name(fld.name)
        rows.append(row)
        print(f"✓ {row['folder']}  ->  {row}")
    except Exception as err:
        print(f"❌ {fld.name}: {err}")

if rows:
    df = pd.DataFrame(rows).sort_values("folder")
    df.to_csv("metrics.csv", index=False)
    print("\nAll done – results written to  metrics.csv")
else:
    print("No folders processed.")

