"""convert all the csv.gz files to parquet for fast loading"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("mimiciii_data_dir", type=Path)
opts = parser.parse_args()

csv_fps = list(opts.mimiciii_data_dir.glob("*.csv.gz"))
for csv_fp in tqdm(csv_fps, unit="file"):
    if csv_fp.with_suffix(".pq").exists():
        continue
    df = pd.read_csv(csv_fp, low_memory=False)
    try:
        df.to_parquet(csv_fp.with_suffix(".pq"), engine="pyarrow", compression="GZIP")
    except (TypeError, ValueError, pyarrow.lib.ArrowTypeError) as e:
        breakpoint()