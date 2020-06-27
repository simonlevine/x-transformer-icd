#!/bin/bash


# DATASET=$1
DATASET = $'mimiciii-14'
# MODEL_TYPE=$2
MAX_XSEQ_LEN= $'128'
#$3 #NEED TO MODIFY THIS? number tokens, = 128 by default. Perhaps too short.

MODEL_TYPE=$'bert'
MODEL_NAME='Bio_ClinicalBERT'

OUTPUT_DIR=save_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
mkdir -p ${PROC_DATA_DIR}
python -u -m xbert.preprocess \
    --do_proc_feat \
    -i ./datasets/${DATASET} \
    -o ${PROC_DATA_DIR} \
    -m ${MODEL_TYPE} \
    -n ${MODEL_NAME} \
    --max_xseq_len ${MAX_XSEQ_LEN} \
    |& tee ${PROC_DATA_DIR}/log.${MODEL_TYPE}.${MAX_XSEQ_LEN}.txt
