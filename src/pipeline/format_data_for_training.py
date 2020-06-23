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


def construct_datasets(icd_version='10', seq_num = '1.0'):
    logger.info("Ingesting MIMIC data...")
    diagnosis_df = pd.read_csv(DIAGNOSIS_CSV_FP, usecols=[
                               "HADM_ID", "ICD9_CODE", "SEQ_NUM"])
                                
    if seq_num == '1.0':
        diagnosis_df = diagnosis_df[diagnosis_df.SEQ_NUM == 1.0]

    icd9_long_description_df = pd.read_csv(ICD9_KEY_FP, usecols=['ICD9_CODE', 'LONG_TITLE'])
    icd_eqivalence_mapping_df = pd.read_csv(ICD_GEM_FP, sep='|', header=None).rename(
        columns={0: 'ICD9_CODE', 1: 'ICD10_CODE', 2: 'LONG_TITLE'})
    icd_eqivalence_mapping_df['ICD9_CODE'] = icd_eqivalence_mapping_df['ICD9_CODE'].str.replace('.', '')
    diagnosis_df = diagnosis_df.merge(
        icd9_long_description_df, left_on=['ICD9_CODE'], right_on=['ICD9_CODE'])


    if icd_version == '10':
        logger.info("Converting ICD9 to ICD10...")
        diagnosis_df = diagnosis_df.merge(
            icd_eqivalence_mapping_df, left_on=['ICD9_CODE'], right_on=['ICD9_CODE'])
        diagnosis_df = diagnosis_df.drop(columns=['ICD9_CODE', 'LONG_TITLE_x']).rename(
            columns={'LONG_TITLE_y': 'LONG_TITLE'})

    note_events_df = pd.read_csv(NOTE_EVENTS_CSV_FP, usecols=[
                                 "HADM_ID", "TEXT", "CATEGORY"])
    note_events_df = note_events_df[note_events_df.CATEGORY ==
                                    "Discharge summary"]
    df_raw = note_events_df.merge(diagnosis_df).dropna()

    logger.info("Pivoting to group related entries and ICD codes...")
    df = df_raw.groupby(["HADM_ID", "TEXT"]).agg({
        ("ICD10_CODE" if icd_version == '10' else "ICD9_CODE"): list,
        "SEQ_NUM": list,
        "LONG_TITLE": list,
    }).reset_index()

    df = df.drop_duplicates('HADM_ID')
    
    logger.info("Splitting into test/train...")
    df_train = df.sample(frac=0.66, random_state=42)
    df_test = df.drop(df_train.index)

    return df_train, df_test


if __name__ == "__main__":
    main()
