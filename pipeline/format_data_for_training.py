from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

DIAGNOSIS_CSV_FP = "./data/mimiciii-14/DIAGNOSES_ICD.csv.gz"
NOTE_EVENTS_CSV_FP = "./data/mimiciii-14/NOTEEVENTS.csv.gz"
ICD9_KEY_FP = "./data/mimiciii-14/D_ICD_DIAGNOSES.csv.gz"
ICD_GEM_FP = "./data/ICD_general_equivalence_mapping.csv"


def load_and_serialize_dataset():
    df_train, df_test = construct_datasets()
    basedir_outpath = Path("./intermediary-data")
    basedir_outpath.mkdir(exist_ok=True)
    for df_, type_ in [(df_train, "train"), (df_test, "test")]:
        fp_out = basedir_outpath/f"notes2diagnosis-icd-{type_}.json.gz"
        logger.info(f"Serializing {type_} dataframe to {fp_out}...")
        df_.to_json(fp_out, orient="split")


def construct_datasets(subsampling = False):
    dataset, _ = load_mimic_dataset()
    dataset = convert_icd9_to_icd10(dataset, load_icd_general_equivalence_mapping())
    df_train, df_test = test_train_validation_split(dataset)

    if subsampling == True:
        logger.info('Subsampling 80 training rows, 20 testing rows of data.')
        df_train = df_train.sample(n=80)
        df_test = df_test.sample(n=20)

    return df_train, df_test


def load_mimic_dataset():
    diagnosis_df = pd.read_csv(DIAGNOSIS_CSV_FP, usecols=["HADM_ID", "ICD9_CODE", "SEQ_NUM"])
    diagnosis_df = diagnosis_df[diagnosis_df.SEQ_NUM==1]
    icd9_long_description_df = pd.read_csv(ICD9_KEY_FP, usecols=["ICD9_CODE", "LONG_TITLE"])
    note_event_cols = ["HADM_ID", "TEXT", "CATEGORY", "ISERROR", "CHARTDATE"]
    note_events_df = pd.read_csv(NOTE_EVENTS_CSV_FP, usecols=note_event_cols)
    note_events_df = note_events_df[note_events_df.CATEGORY == "Discharge summary"]
    note_events_df = note_events_df.drop_duplicates(["TEXT"])
    #filtering out dates, etc.
    note_events_df.text = note_events_df.text.str.replace('\[.*?\]', '', regex=True)

    full_df = note_events_df.merge(diagnosis_df.merge(icd9_long_description_df)).drop_duplicates(["HADM_ID"])
    full_df = full_df[["HADM_ID", "TEXT", "SEQ_NUM", "ICD9_CODE", "LONG_TITLE"]]
    return full_df, (diagnosis_df, icd9_long_description_df, note_events_df)


def load_icd_general_equivalence_mapping():
    icd_equiv_map_df = pd.read_csv(
        ICD_GEM_FP,
        sep="|",
        header=None,
        names=["ICD9_CODE", "ICD10_CODE", "LONG_TITLE_ICD10"]
    )
    icd_equiv_map_df = icd_equiv_map_df.dropna() # there is a single blank line
    icd_equiv_map_df["ICD9_CODE"] = \
        icd_equiv_map_df["ICD9_CODE"].str.replace('.', '')
    return icd_equiv_map_df.groupby("ICD9_CODE").agg(set)


def convert_icd9_to_icd10(dataset: pd.DataFrame, equivalence_mapping: pd.DataFrame):
    return dataset \
        .merge(equivalence_mapping, left_on=["ICD9_CODE"], right_index=True) \
        .rename(columns={"LONG_TITLE": "LONG_TITLE_ICD9"})


def test_train_validation_split(dataset):
    df_train = dataset.sample(frac=0.66, random_state=42)
    df_test = dataset.drop(df_train.index)
    return df_train, df_test


if __name__ == "__main__":
    load_and_serialize_dataset()
