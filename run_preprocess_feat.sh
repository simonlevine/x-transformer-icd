#!/usr/bin/env bash

source params.sh

DATASET=$'mimiciii-14'

MAX_XSEQ_LEN=$(params "['max_seq_len']")
MAX_XCHAR_LEN=$(params "['max_char_len']")
MODEL_TYPE=$'longformer'
MODEL_NAME=$(params "['model_name']")

DATA_DIR=./data/intermediary-data
OUTPUT_DIR=${DATA_DIR}/xbert_outputs
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data

mkdir -p ${PROC_DATA_DIR}
$PY_CONDA xbert/preprocess.py \
    --do_proc_feat \
    -i ${DATA_DIR}/xbert_inputs \
    -o ${PROC_DATA_DIR} \
    -m ${MODEL_TYPE} \
    -n ${MODEL_NAME} \
    --max_xseq_len ${MAX_XSEQ_LEN} #\
    # --max_trunc_char ${MAX_XCHAR_LEN}
    # |& tee ${PROC_DATA_DIR}/log.${MODEL_TYPE}.${MAX_XSEQ_LEN}.txt
