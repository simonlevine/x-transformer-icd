#!/bin/bash

DATASET=$'mimiciii-14'
DATA_DIR=../../data/intermediary-data/xbert_inputs

LABEL_NAME = $'pifa-tfidf-s0'
MODEL_NAME = $'emilyalsentzer/Bio_ClinicalBERT'
EXP_NAME=${DATASET}.final

PRED_NPZ_PATHS=""
OUTPUT_DIR=saved_models/${DATASET}/${LABEL_NAME}
INDEXER_DIR=${OUTPUT_DIR}/indexer
MATCHER_DIR=${OUTPUT_DIR}/matcher/${MODEL_NAME}
RANKER_DIR=${OUTPUT_DIR}/ranker/${MODEL_NAME}
mkdir -p ${RANKER_DIR}

# train linear ranker
python -m xbert.ranker train \
    -x1 ${DATA_DIR}/X.trn.npz \
    -x2 ${MATCHER_DIR}/trn_embeddings.npy \
    -y ${DATA_DIR}/Y.trn.npz \
    -z ${MATCHER_DIR}/C_trn_pred.npz \
    -c ${INDEXER_DIR}/code.npz \
    -o ${RANKER_DIR} -t 0.01 \
    -f 0 -ns 2 --mode ranker

# predict final label ranking
PRED_NPZ_PATH=${RANKER_DIR}/tst.pred.npz
python -m xbert.ranker predict \
    -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
    -x1 ${DATA_DIR}/X.tst.npz \
    -x2 ${MATCHER_DIR}/tst_embeddings.npy \
    -y ${DATA_DIR}/Y.tst.npz \
    -z ${MATCHER_DIR}/C_tst_pred.npz \
    -f 0 -t noop

# append the prediction path (changed to just one here.)
PRED_NPZ_PATHS="${PRED_NPZ_PATHS} ${PRED_NPZ_PATH}"

# final eval
EVAL_DIR=results_transformer-large
mkdir -p ${EVAL_DIR}
python -u -m xbert.evaluator \
    -y ${DATA_DIR}/Y.tst.npz \
    -e -p ${PRED_NPZ_PATHS} \
    |& tee ${EVAL_DIR}/${EXP_NAME}.txt

