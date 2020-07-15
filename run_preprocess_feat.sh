#!/bin/bash

# DATASET=$1
DATASET=$'mimiciii-14'
# MODEL_TYPE=$2
MAX_XSEQ_LEN=$'128'
#$3 #NEED TO MODIFY THIS? number tokens, = 128 by default. Perhaps too short.

MODEL_TYPE=$'bert'
MODEL_NAME=$'emilyalsentzer/Bio_ClinicalBERT'

DATA_DIR=../../data/intermediary-data
OUTPUT_DIR=${DATA_DIR}/xbert_outputs
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data

source create_conda_env_as_necessary.sh

mkdir -p ${PROC_DATA_DIR}
python -u -m xbert.preprocess \
    --do_proc_feat \
    -i ${DATA_DIR}/xbert_inputs/${DATASET} \
    -o ${PROC_DATA_DIR} \
    -m ${MODEL_TYPE} \
    -n ${MODEL_NAME} \
    --max_xseq_len ${MAX_XSEQ_LEN} \
    |& sudo tee ${PROC_DATA_DIR}/log.${MODEL_TYPE}.${MAX_XSEQ_LEN}.txt