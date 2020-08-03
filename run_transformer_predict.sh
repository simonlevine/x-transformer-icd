#!/usr/bin/env bash

source params.sh

DATASET=$'mimiciii-14'
DATA_DIR=./data/intermediary-data/xbert_inputs

LABEL_NAME=$'pifa-tfidf-s0'
MODEL_NAME=$(params "['model_name']")
MODEL_FOLDER_NAME='roberta'
EXP_NAME=${DATASET}.final

OUTPUT_DIR=./data/intermediary-data/xbert_outputs/${LABEL_NAME}
INDEXER_DIR=${OUTPUT_DIR}/indexer
MATCHER_DIR=${OUTPUT_DIR}/matcher
RANKER_DIR=${OUTPUT_DIR}/ranker
mkdir -p ${RANKER_DIR}

# train linear ranker
$PY_CONDA xbert/ranker.py train \
    -x1 ${DATA_DIR}/X.trn.npz \
    -x2 ${MATCHER_DIR}/trn_embeddings.npy \
    -y ${DATA_DIR}/Y.trn.npz \
    -z ${MATCHER_DIR}/C_trn_pred.npz \
    -c ${INDEXER_DIR}/code.npz \
    -o ${RANKER_DIR} -t 0.01 \
    -f 0 -ns 2 --mode ranker

# predict final label ranking
PRED_NPZ_PATH=${RANKER_DIR}/tst.pred.npz

$PY_CONDA xbert/ranker.py predict \
    -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
    -x1 ${DATA_DIR}/X.tst.npz \
    -x2 ${MATCHER_DIR}/tst_embeddings.npy \
    -y ${DATA_DIR}/Y.tst.npz \
    -z ${MATCHER_DIR}/C_tst_pred.npz \
    -f 0 -t noop

# final eval
EVAL_DIR=results_transformer-large
mkdir -p ${EVAL_DIR}
$PY_CONDA xbert/evaluator.py \
    -y ${DATA_DIR}/Y.tst.npz \
    -e -p ${PRED_NPZ_PATH} \
    |& tee ${EVAL_DIR}/${EXP_NAME}.txt

