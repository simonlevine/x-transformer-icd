from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

DIAGNOSIS_CSV_FP = "../data/mimiciii-14/DIAGNOSES_ICD.csv.gz"
NOTE_EVENTS_CSV_FP = "../data/mimiciii-14/NOTEEVENTS.csv.gz"
ICD9_KEY_FP = "../data/mimiciii-14/D_ICD_DIAGNOSES.csv.gz"
ICD_GEM_FP = "../data/ICD_general_equivalence_mapping.csv"

def main():
    df_train, df_test = construct_datasets()
    (basedir_outpath := Path("./intermediary-data")).mkdir(exist_ok=True)
    for df_, type_ in [(df_train, "train"), (df_test, "test")]:
        fp_out = basedir_outpath/f"notes2diagnosis-icd-{type_}.json.gz"
        logger.info(f"Serializing {type_} dataframe to {fp_out}...")
        df_.to_json(fp_out, orient="split")


def construct_datasets(icd_version='10'):
    logger.info("Ingesting MIMIC data...")
    diagnosis_df = pd.read_csv(DIAGNOSIS_CSV_FP, usecols=[
                               "HADM_ID", "ICD9_CODE", "SEQ_NUM"])
    # merging with description...
    icd9_key_df = pd.read_csv(ICD9_KEY_FP, usecols=['ICD9_CODE', 'LONG_TITLE'])
    #pulling general equivalence mapping...
    icd_gem_df = pd.read_csv(ICD_GEM_FP, sep='|', header=None).rename(
        columns=dict(zip((0, 1, 2), ('ICD9_CODE', 'ICD10_CODE', 'LONG_TITLE'))))

    icd_gem_df['ICD9_CODE'] = icd_gem_df['ICD9_CODE'].str.replace('.', '')

    diagnosis_df = diagnosis_df.merge(
        icd_key_df, left_on=['ICD9_CODE'], right_on=['ICD9_CODE'])

    if icd_version == '10':
        logger.info("Converting ICD9 to ICD10...")

        diagnosis_df = diagnosis_df.merge(
            icd_gem, left_on=['ICD9_CODE'], right_on=['ICD9_CODE'])
        diagnosis_df = diagnosis_df.drop(columns=['ICD9_CODE', 'LONG_TITLE_x']).rename(
            columns={'LONG_TITLE_y': 'LONG_TITLE'})

    note_events_df = pd.read_csv(NOTE_EVENTS_CSV_FP, usecols=[
                                 "HADM_ID", "TEXT", "CATEGORY"])
    note_events_df = note_events_df[note_events_df.CATEGORY ==
                                    "Discharge summary"]
    df_raw = note_events_df.merge(diagnosis_df).dropna()
    logger.info("Pivoting to group related entries and ICD codes...")

    if icd_version == '10':
        df = df_raw.groupby(["HADM_ID", "TEXT"]).agg({
            "ICD10_CODE": set,
            "SEQ_NUM": set,
            "LONG_TITLE": set,
        }).reset_index()

    else:
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
