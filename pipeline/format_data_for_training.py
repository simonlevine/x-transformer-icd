from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from loguru import logger

DIAGNOSIS_CSV_FP = "./data/mimiciii-14/DIAGNOSES_ICD.csv.gz"
PROCEDURE_CSV_FP = "./data/mimiciii-14/PROCEDURES_ICD.csv"
NOTE_EVENTS_CSV_FP = "./data/mimiciii-14/NOTEEVENTS.csv.gz"
ICD9_DIAG_KEY_FP = "./data/mimiciii-14/D_ICD_DIAGNOSES.csv.gz"
ICD9_PROC_KEY_FP = "./data/mimiciii-14/D_ICD_PROCEDURES.csv"
ICD_GEM_FP = "./data/ICD_general_equivalence_mapping.csv"


with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f.read())
icd_version_specified = str(params['prepare_for_xbert']['icd_version'])
diag_or_proc_param = params['prepare_for_xbert']['diag_or_proc']
assert diag_or_proc_param == 'proc' or diag_or_proc_param == 'diag', 'Must specify either \'proc\' or \'diag\'.'
note_category_param = params['prepare_for_xbert']['note_category']
icd_seq_num_param = params['prepare_for_xbert']['one_or_all_icds']
subsampling_param = params['prepare_for_xbert']['subsampling']

def load_and_serialize_dataset():
    df_train, df_test = construct_datasets(
        diag_or_proc_param, note_category_param, subsampling_param)
    basedir_outpath = Path("./intermediary-data")
    basedir_outpath.mkdir(exist_ok=True)
    for df_, type_ in [(df_train, "train"), (df_test, "test")]:
        fp_out = basedir_outpath/f"notes2diagnosis-icd-{type_}.json.gz"
        logger.info(f"Serializing {type_} dataframe to {fp_out}...")
        df_.to_json(fp_out, orient="split")


def construct_datasets(diag_or_proc_param, note_category_param, subsampling_param):
    dataset, _ = load_mimic_dataset(
        diag_or_proc_param, note_category_param, icd_seq_num_param)
    if icd_seq_num_param == '10':
        dataset = convert_icd9_to_icd10(dataset, load_icd_general_equivalence_mapping())

    df_train, df_test = test_train_validation_split(dataset)
    if subsampling_param == True:
        logger.info('Subsampling 80 training rows, 20 testing rows of data.')
        df_train = df_train.sample(n=80)
        df_test = df_test.sample(n=20)

    return df_train, df_test


def load_mimic_dataset(diag_or_proc_param, note_category_param, icd_seq_num_param):

    logger.info(f'Loading notes from {note_category_param} category...')
    note_event_cols = ["HADM_ID", "TEXT", "CATEGORY", "ISERROR", "CHARTDATE"]
    note_events_df = pd.read_csv(NOTE_EVENTS_CSV_FP, usecols=note_event_cols)
    note_events_df = note_events_df[note_events_df.CATEGORY ==
                                    note_category_param]


    if diag_or_proc_param == 'diag':
        logger.info('Loading diagnosis outcome data...')
        diag_df = pd.read_csv(DIAGNOSIS_CSV_FP, usecols=[
            "HADM_ID", "ICD9_CODE", "SEQ_NUM"])
        icd9_long_description_df = pd.read_csv(
            ICD9_DIAG_KEY_FP, usecols=["ICD9_CODE", "LONG_TITLE"])
        full_df = note_events_df.merge(diag_df.merge(
            icd9_long_description_df))
        full_df = full_df[["HADM_ID", "TEXT",
                            "CATEGORY", "SEQ_NUM", "ICD9_CODE", "LONG_TITLE"]]

    elif diag_or_proc_param == 'proc':
        logger.info('Loading procedure outcome data...')
        proc_df = pd.read_csv(PROCEDURE_CSV_FP, usecols=[
            "HADM_ID", "ICD9_CODE", "SEQ_NUM"])
        icd9_long_description_df = pd.read_csv(
            ICD9_PROC_KEY_FP, usecols=["ICD9_CODE", "LONG_TITLE"])
        full_df = note_events_df.merge(proc_df.merge(
            icd9_long_description_df))
        full_df = full_df[["HADM_ID", "TEXT",
                                 "CATEGORY", "SEQ_NUM", "ICD9_CODE", "LONG_TITLE"]]
                                 
    logger.info(f'Setting included ICD sequence number to {icd_seq_num_param}')
    if icd_seq_num_param != 'all':
        full_df = full_df[full_df.SEQ_NUM == icd_seq_num_param]
    
    return full_df, (icd9_long_description_df, note_events_df)


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
