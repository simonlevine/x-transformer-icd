"""
XBERT PREPROCESSING (mimic_iii_1-4)

This module preprocesses train/test dataframes generated using
format_data_for_training.py (assuming MIMICiii, with ICD10s converted,
and only SEQ_NUM = 1.0) in preparation for the XBERT pipeline.

Running this script produces:
X.trn.npz: the instance TF-IDF feature matrix for the train set.
    The data type is scipy.sparse.csr_matrix of size (N_trn, D_tfidf),
    where N_trn is the number of train instances and D_tfidf is the number of features.

X.tst.npz: the instance TF-IDF feature matrix for the test set.
    The data type is scipy.sparse.csr_matrix of size (N_tst, D_tfidf),
    where N_tst is the number of test instances
    and D_tfidf is the number of features.

Y.trn.npz: the instance-to-label matrix for the train set.
    The data type is scipy.sparse.csr_matrix of size (N_trn, L),
    where n_trn is the number of train instances and L is the number of labels.

Y.tst.npz: the instance-to-label matrix for the test set.
    The data type is scipy.sparse.csr_matrix of size (N_tst, L),
    where n_tst is the number of test instances and L is the number of labels.

train_raw_texts.txt: The raw text of the train set.

test_raw_texts.txt: The raw text of the test set.

label_map.txt: the label's text description.

-----
Next, these files should be places in the proper {DATASET} folder for xbert.
Given the input files, the XBERT pipeline (Indexer, Matcher, and Ranker) can then be run downstream.
"""

import typing as t
import re
import pickle
import numpy as np
import pandas as pd
import scipy
import yaml
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

try:
    import format_data_for_training #script from auto-icd
except ImportError:
    # when running in a pytest context
    from . import format_data_for_training

# input filepaths.
DIAGNOSIS_CSV_FP = "./data/mimiciii-14/DIAGNOSES_ICD.csv.gz"
PROCEDURES_CSV_FP = "./data/mimiciii-14/PROCEDURES_ICD.csv.gz"
ICD9_DIAG_KEY_FP = "./data/mimiciii-14/D_ICD_DIAGNOSES.csv.gz"
ICD9_PROC_KEY_FP = "./data/mimiciii-14/D_ICD_DIAGNOSES.csv.gz"
ICD_GEM_FP = "./data/ICD_general_equivalence_mapping.csv"

# output filepaths
XBERT_LABEL_MAP_FP = './data/intermediary-data/xbert_inputs/label_map.txt'
XBERT_TRAIN_RAW_TEXTS_FP = './data/intermediary-data/xbert_inputs/train_raw_texts.txt'
XBERT_TEST_RAW_TEXTS_FP = './data/intermediary-data/xbert_inputs/test_raw_texts.txt'
XBERT_X_TRN_FP = './data/intermediary-data/xbert_inputs/X.trn.npz'
XBERT_X_TST_FP = './data/intermediary-data/xbert_inputs/X.tst.npz'
XBERT_Y_TRN_FP = './data/intermediary-data/xbert_inputs/Y.trn.npz'
XBERT_Y_TST_FP = './data/intermediary-data/xbert_inputs/Y.tst.npz'
DF_TRAIN_FP ='./data/intermediary-data/df_train.pkl'
DF_TEST_FP = './data/intermediary-data/df_test.pkl'
TF_IDF_VECTORIZER_PICKLE_FP = './data/model-artifacts/tf-idf-vectorizor.pkl'


def main():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f.read())
    icd_version_specified = str(params['prepare_for_xbert']['icd_version'])
    diag_or_proc_param = params['prepare_for_xbert']['diag_or_proc']
    assert diag_or_proc_param == 'proc' or diag_or_proc_param == 'diag', 'Must specify either \'proc\' or \'diag\'.'
    note_category_param = params['prepare_for_xbert']['note_category']
    icd_seq_num_param = params['prepare_for_xbert']['one_or_all_icds']
    subsampling_param = params['prepare_for_xbert']['one_or_all_icds']

    logger.info(f'Using ICD version {icd_version_specified}...')
    assert icd_version_specified == '9' or icd_version_specified == '10', 'Must specify one of ICD9 or ICD10.'
    logger.info('reformatting raw data with subsampling {}', 'enabled' if subsampling_enabled_param else 'disabled')

    df_train, df_test = \
        format_data_for_training.construct_datasets(
            diag_or_proc_param, note_category_param, subsampling_param)

    X_trn = xbert_prepare_txt_inputs(df_train, 'training')
    X_tst = xbert_prepare_txt_inputs(df_test, 'testing')
    X_trn_tfidf, X_tst_tfidf = xbert_get_tfidf_inputs(X_trn, X_tst)
    icd_labels, desc_labels = xbert_create_label_map(
        icd_version_specified, diag_or_proc_param)
    desc_labels = desc_labels.apply(xbert_clean_label)

    Y_trn_map = xbert_prepare_Y_maps(
        df_train, icd_labels.tolist(), icd_version_specified)
    Y_tst_map = xbert_prepare_Y_maps(
        df_test, icd_labels.tolist(), icd_version_specified)

    xbert_write_preproc_data_to_file(
        desc_labels, X_trn, X_tst, X_trn_tfidf, X_tst_tfidf, Y_trn_map, Y_tst_map)

    logger.info(
        'Done preprocessing. Saving pickled dataframes to file for later postprocessing.'
    )
    df_train.to_pickle(DF_TRAIN_FP)
    df_test.to_pickle(DF_TEST_FP)
    with open(TF_IDF_VECTORIZER_PICKLE_FP, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info('Done.')


def xbert_clean_label(label):
    return re.sub(r"[,.:;\\''/@#?!\[\]&$_*]+", ' ', label).strip()


def xbert_create_label_map(icd_version, diag_or_proc_param):
    """creates a dataframe of all ICD9 or ICD10 labels (CM or PCS) and corresponding
    descriptions in 2018 ICD code set (if 10).
    Note that this is inclusive of, but not limited to,
    the set of codes appearing in cases of MIMIC-iii."""

    logger.info(
        f'Creating ICD {icd_version} and long title lists for xbert...')

    ##TODO: this block should be imported from format data

    # use general equivalnce mapping to create label map.
    if icd_version == '10':
        assert diag_or_proc_param == 'diag', 'Cannot currently process ICD10-PCS labels.'

        icd_equivalence_df = pd.read_csv(ICD_GEM_FP, sep='|', header=None).rename(columns=dict(zip(
            (1, 2), ('ICD10_CODE', 'LONG_TITLE')))).drop(0, axis=1).drop_duplicates().reset_index(drop=True).dropna()
        desc_labels = icd_equivalence_df['LONG_TITLE']
        assert desc_labels.shape == desc_labels.dropna().shape
        icd_labels = icd_equivalence_df['ICD10_CODE']
        assert icd_labels.shape == icd_labels.dropna().shape

    elif icd_version == '9':  # use icd9 labels directly from mimic dataset.
        if diag_or_proc_param == 'diag':
            icd9_df = pd.read_csv(ICD9_DIAG_KEY_FP, usecols=['ICD9_CODE', 'LONG_TITLE'])
        elif diag_or_proc_param == 'proc':
            icd9_df = pd.read_csv(ICD9_PROC_KEY_FP, usecols=['ICD9_CODE', 'LONG_TITLE'])
        desc_labels = icd9_df['LONG_TITLE']
        assert desc_labels.shape == desc_labels.dropna().shape
        icd_labels = icd9_df['ICD9_CODE']
        assert icd_labels.shape == icd_labels.dropna().shape

    return icd_labels, desc_labels


def xbert_prepare_Y_maps(df, icd_labels, icd_version):
    """Creates a binary mapping of
    icd labels to appearance in a patient account
    (icd to hadm_id)
    
    Args:
        df (DataFrame): training or testing dataframe.
        df_subset (str): "train", "test", or "validation"
        icd_labels (List[str]): Series of all possible icd labels
    
    Returns:
        Y_: a binary DataFrame of size N by K, where
            N is the number of samples (HADM_IDs) in the
            train or test dataframe, and K is the number
            of potential ICD labels."""
    hadm_ids = df.HADM_ID.unique().tolist()
    Y_ = pd.DataFrame(index=hadm_ids, columns=icd_labels)
    with tqdm(total=len(df), unit="HADM id") as pbar:
        for idx, row in df.iterrows():
            if icd_version == '10':
                Y_.loc[row.HADM_ID, row.ICD10_CODE] = 1
            elif icd_version == '9':
                Y_.loc[row.HADM_ID, row.ICD9_CODE] = 1
            pbar.update(1)

    return Y_.fillna(0)


def xbert_prepare_txt_inputs(df, df_subset):
    logger.info(
        f'Collecting {df_subset} free-text as input features to X-BERT...')
    raw_texts = df[['TEXT']].replace(r'\n', ' ', regex=True)  # train stage expects each example to fit on a single line
    return raw_texts


def xbert_get_tfidf_inputs(X_trn, X_tst, n_gram_range_upper=1, min_doc_freq = 1):
    """
    Creates tf-idf vectors of instances in preparation for xbert training.
    """

    logger.info('Creating TF_IDF inputs...')

    vectorizer = TfidfVectorizer(
        ngram_range=(1, n_gram_range_upper),
        min_df=min_doc_freq)

    logger.info('Fitting vectorizers to corpora...')

    corpus_trn = list(X_trn.values.flatten())
    corpus_tst = list(X_tst.values.flatten())

    logger.info('TF-IDF Vectorizing training text samples...')
    X_trn_tfidf = vectorizer.fit_transform(corpus_trn)
    logger.info('TF-IDF Vectorizing testing text samples...')
    X_tst_tfidf = vectorizer.transform(corpus_tst)
    return X_trn_tfidf, X_tst_tfidf

def xbert_write_preproc_data_to_file(desc_labels, X_trn, X_tst, X_trn_tfidf, X_tst_tfidf, Y_trn, Y_tst):
    """Creates X_trn/X_tst TF-IDF vectors, (csr/npz files),
    Y_trn/Y_tst (binary array; csr/npz files), as well as
    .txt files for free text labels (label_map.txt) and train/test inputs (train/test_raw_texts)
    in preparation for XBERT training."""

    #writing label map (icd descriptions) to txt
    logger.info('Writing icd LONG_TITLE (label map) to txt.')
    desc_labels.to_csv(path_or_buf=XBERT_LABEL_MAP_FP,
                      header=None, index=None, sep='\t', mode='w')

    #writing raw text features to txts
    logger.info('Writing data raw features to txt.')
    X_trn.to_csv(path_or_buf=XBERT_TRAIN_RAW_TEXTS_FP,
                 header=None, index=None, sep='\t', mode='w')
    X_tst.to_csv(path_or_buf=XBERT_TEST_RAW_TEXTS_FP,
                 header=None, index=None, sep='\t', mode='w')

    #writing X.trn.npz, X.tst.npz files.
    logger.info(
        'Saving TFIDF of features (sparse compressed row matrices) to file...')
    scipy.sparse.save_npz(XBERT_X_TRN_FP, X_trn_tfidf)
    scipy.sparse.save_npz(XBERT_X_TST_FP, X_tst_tfidf)

    #writing Y.trn.npz and Y.tst.npz to file.
    logger.info(
        'Saving binary label occurrence array (sparse compressed row) to file...')
    Y_trn_csr = scipy.sparse.csr_matrix(Y_trn.values)
    Y_tst_csr = scipy.sparse.csr_matrix(Y_tst.values)
    scipy.sparse.save_npz(XBERT_Y_TRN_FP, Y_trn_csr)
    scipy.sparse.save_npz(XBERT_Y_TST_FP, Y_tst_csr)

    logger.info('Done.')


if __name__ == "__main__":
    main()
