import numpy as np
import pandas as pd
import scipy
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy

import format_data_for_training


DIAGNOSIS_CSV_FP = "../data/mimiciii-14/DIAGNOSES_ICD.csv.gz"
ICD9_KEY_FP = "../data/mimiciii-14/D_ICD_DIAGNOSES.csv.gz"
ICD_GEM_FP = "../data/ICD_general_equivalence_mapping.csv"

XBERT_LABEL_MAP_FP = '../data/xbert_inputs/label_map.txt'

XBERT_TRAIN_RAW_LABELS_FP = '../data/xbert_inputs/train_raw_labels.txt'
XBERT_TEST_RAW_LABELS_FP = '../data/xbert_inputs/test_raw_labels.txt'

XBERT_X_TRN_FP = '../data/xbert_inputs/X.trn.npz'
XBERT_X_TST_FP = '../data/xbert_inputs/X.tst.npz'

XBERT_Y_TRN_FP = '../data/xbert_inputs/Y.trn.npz'
XBERT_Y_TST_FP = '../data/xbert_inputs/Y.tst.npz'


def main():
    df_train, df_test = format_data_for_training.construct_datasets()
    #DUPLICATES PULL IN!
    df_test.drop_duplicates('HADM_ID')
    df_train.drop_duplicates('HADM_ID')

    X_trn, X_tst = xbert_prepare_txt_inputs(
        df_train, 'training'), xbert_prepare_txt_inputs(df_test, 'testing')
    X_trn_tfidf, X_tst_tfidf = xbert_get_tfidf_inputs(X_trn, X_tst)
    icd_labels, desc_labels = xbert_create_label_map(icd_version='10')
    Y_trn_map, Y_tst_map = xbert_prepare_Y_maps(
        df_train, 'training', icd_labels), xbert_prepare_Y_maps(df_test, 'testing', icd_labels)

    xbert_write_preproc_data_to_file(
        icd_labels, X_trn, X_tst, X_trn_tfidf, X_tst_tfidf, Y_trn_map, Y_tst_map)


def xbert_create_label_map(icd_version='10'):
    """creates a dataframe of all ICD10 labels and corresponding
    descriptions in 2018 ICD code set.
    Note that this is inclusive of, but not limited to,
    the set of codes appearing in cases of MIMIC-iii. """

    assert icd_version == '10', 'Only ICD10 is currently supported.'

    logger.info('Creating ICD and long title lists for xbert...')
    icd_equivalence_df = pd.read_csv(ICD_GEM_FP, sep='|', header=None).rename(columns=dict(zip(
        (1, 2), ('ICD10_CODE', 'LONG_TITLE')))).drop(0, axis=1).drop_duplicates().reset_index(drop=True)

    desc_labels = icd_equivalence_df['LONG_TITLE']
    icd_labels = icd_equivalence_df['ICD10_CODE']
    return icd_labels, desc_labels


def xbert_prepare_Y_maps(df, df_subset, icd_labels):
    """Creates a binary mapping of
    labels to appearance"""

    df = df.set_index('HADM_ID')
    Y_ = pd.DataFrame(np.zeros((df.shape[0], icd_labels.shape[0])))
    Y_.columns = icd_labels
    Y_.index = df.index
    logger.info(
        f'Constructing label mapping ({df_subset} portion) ICD10 codes to HADM_ID...')

    for hadm_id in Y_.index:  # running through rows.
        curr_primary_icd = df.loc[hadm_id, 'ICD10_CODE'][0]
        Y_.loc[hadm_id, curr_primary_icd] += 1
    return Y_


def xbert_prepare_txt_inputs(df, df_subset):
    logger.info(
        f'Collecting {df_subset} free-text as input features to X-BERT...')
    return df[['TEXT']]



def xbert_get_tfidf_inputs(X_trn, X_tst):
    logger.info('Creating TF_IDF inputs...')
    vect = TfidfVectorizer()
    corpus = list(X_trn.values.flatten())
    logger.info('Fitting a vectorizer to the training data corpus...')
    fitted_vect = vect.fit(corpus)  # TRAINING DATA ONLY for corpus
    logger.info('TF-IDF Vectorizing training text samples...')
    X_trn_tfidf = fitted_vect.transform(X_trn.values.flatten())
    logger.info('TF-IDF Vectorizing testing text samples...')
    X_tst_tfidf = fitted_vect.transform(X_tst.values.flatten())

    return X_trn_tfidf, X_tst_tfidf


def xbert_write_preproc_data_to_file(icd_labels, X_trn, X_tst, X_trn_tfidf, X_tst_tfidf, Y_trn, Y_tst):
    """Creates X_trn/X_tst TF-IDF vectors, (csr/npz files),
    Y_trn/Y_tst (binary array; csr/npz files), as well as
    .txt files for free text labels (label_map.txt) and train/test inputs (train/test_raw_texts)
    in preparation for XBERT training."""

    #writing label map (icd descriptions) to txt
    logger.info('Writing icd LONG_TITLE (label map) to txt.')
    icd_labels.to_csv(path_or_buf=XBERT_LABEL_MAP_FP,
                      header=None, index=None, sep='\t', mode='a')

    #writing raw text features to txts
    logger.info('Writing data raw features to txt.')
    X_trn.to_csv(path_or_buf=XBERT_TRAIN_RAW_LABELS_FP,
                 header=None, index=None, sep='\t', mode='a')
    X_tst.to_csv(path_or_buf=XBERT_TEST_RAW_LABELS_FP,
                 header=None, index=None, sep='\t', mode='a')

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
