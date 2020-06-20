from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

DIAGNOSIS_CSV_FP = "../data/mimiciii-14/DIAGNOSES_ICD.csv.gz"
NOTE_EVENTS_CSV_FP = "../data/mimiciii-14/NOTEEVENTS.csv.gz"


def main():
    df_train, df_test = construct_datasets()
    (basedir_outpath := Path("./intermediary-data")).mkdir(exist_ok=True)
    for df_, type_ in [(df_train, "train"), (df_test, "test")]:
        fp_out = basedir_outpath/f"notes2diagnosis-icd-{type_}.json.gz"
        logger.info(f"Serializing {type_} dataframe to {fp_out}...")
        df_.to_json(fp_out, orient="split")


def construct_datasets():
    logger.info("Ingesting MIMIC data...")
    diagnosis_df = pd.read_csv(DIAGNOSIS_CSV_FP, usecols=[
                               "HADM_ID", "ICD9_CODE", "SEQ_NUM"])
    # merging with description...
    icd_key_df = pd.read_csv(ICD_KEY_FP, usecols=['ICD9_CODE', 'LONG_TITLE'])
    diagnosis_df = diagnosis_df.merge(
        icd_key_df, left_on=['ICD9_CODE'], right_on=['ICD9_CODE'])
    note_events_df = pd.read_csv(NOTE_EVENTS_CSV_FP, usecols=[
                                 "HADM_ID", "TEXT", "CATEGORY"])
    note_events_df = note_events_df[note_events_df.CATEGORY ==
                                    "Discharge summary"]
    df_raw = note_events_df.merge(diagnosis_df).dropna()
    logger.info("Pivoting to group related entries and ICD codes...")
    df = df_raw.groupby(["HADM_ID", "TEXT"]).agg({
        "ICD9_CODE": set,
        "SEQ_NUM": set,
        "LONG_TITLE": set,
    }).reset_index()
    logger.info("Splitting into test/train...")
    df_train = df.sample(frac=0.66, random_state=42)
    df_test = df.drop(df_train.index)

    return df_train, df_test


if __name__ == "__main__":
    main()
