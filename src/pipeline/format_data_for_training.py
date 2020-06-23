from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

DIAGNOSIS_CSV_FP = "../data/mimiciii-14/DIAGNOSES_ICD.csv.gz"
NOTE_EVENTS_CSV_FP = "../data/mimiciii-14/NOTEEVENTS.csv.gz"
ICD9_KEY_FP = "../data/mimiciii-14/D_ICD_DIAGNOSES.csv.gz"
ICD_GEM_FP = "../data/ICD_general_equivalence_mapping.csv"

def load_and_serialize_dataset():
    df_train, df_test = construct_dataset()
    (basedir_outpath := Path("./intermediary-data")).mkdir(exist_ok=True)
    for df_, type_ in [(df_train, "train"), (df_test, "test")]:
        fp_out = basedir_outpath/f"notes2diagnosis-icd-{type_}.json.gz"
        logger.info(f"Serializing {type_} dataframe to {fp_out}...")
        df_.to_json(fp_out, orient="split")


def construct_dataset():
    dataset, _ = load_mimic_dataset()
    dataset, _ = convert_icd9_to_icd10(dataset)
    dataset_related_grouped = group_related_entries(dataset)
    df_train, df_test = test_train_validation_split(dataset_related_grouped)
    return df_train, df_test


def load_mimic_dataset():
    diagnosis_df = pd.read_csv(DIAGNOSIS_CSV_FP, usecols=["HADM_ID", "ICD9_CODE", "SEQ_NUM"])
    icd9_long_description_df = pd.read_csv(ICD9_KEY_FP, usecols=["ICD9_CODE", "LONG_TITLE"])
    note_events_df = pd.read_csv(NOTE_EVENTS_CSV_FP, usecols=["HADM_ID", "TEXT", "CATEGORY", "ISERROR"])
    note_events_df = note_events_df[note_events_df.CATEGORY == "Discharge summary"]
    note_events_df = note_events_df.drop_duplicates(["HADM_ID", "TEXT"])
    full_df = note_events_df.merge(diagnosis_df.merge(icd9_long_description_df))
    full_df = full_df[["HADM_ID", "TEXT", "SEQ_NUM", "ICD9_CODE", "LONG_TITLE"]]
    return full_df, (diagnosis_df, icd9_long_description_df, note_events_df)


def convert_icd9_to_icd10(dataset):
    icd_general_eqivalence_mapping_df = \
        pd.read_csv(ICD_GEM_FP, sep='|', header=None, names=["ICD9_CODE", "ICD10_CODE", "LONG_TITLE_ICD10"])
    icd_general_eqivalence_mapping_df["ICD9_CODE"] = \
        icd_general_eqivalence_mapping_df["ICD9_CODE"].str.replace('.', '')
    dataset = (dataset
               .merge(icd_general_eqivalence_mapping_df, left_on=['ICD9_CODE'], right_on=['ICD9_CODE'])
               .rename(columns={"LONG_TITLE": "LONG_TITLE_ICD9"}))
    return dataset, icd_general_eqivalence_mapping_df


def group_related_entries(dataset):
    return dataset.groupby(["HADM_ID", "TEXT"]).agg({
        "ICD9_CODE": set,
        "ICD10_CODE": set,
        "SEQ_NUM": set,
        "LONG_TITLE": set,
    }).reset_index()


def test_train_validation_split(dataset):
    df_train = dataset.sample(frac=0.66, random_state=42)
    df_test = dataset.drop(df_train.index)
    return df_train, df_test


if __name__ == "__main__":
    load_and_serialize_dataset()
