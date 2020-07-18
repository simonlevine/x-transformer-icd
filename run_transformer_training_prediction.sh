#!/usr/bin/env bash

source create_conda_env_as_necessary.sh
source params.sh

NVIDIA_VISIBLE_DEVICES=0

GPID=${0} #CUDA visible devices. 0 for just one 2070 GPU.
# NPROC_PER_NODE='1'

DATASET=$'mimiciii-14'
INDEXER_NAME=$'pifa-tfidf-s0' # or ||| pifa-neural-s0 ||| text-emb-s0

MODEL_NAME='emilyalsentzer/Bio_ClinicalBERT'
MODEL_FOLDER_NAME='Bio_ClinicalBERT'
MODEL_TYPE=$'bert'

OUTPUT_DIR=../../data/intermediary-data/xbert_outputs
PROC_DATA_DIR=../../data/intermediary-data/xbert_outputs/proc_data
MAX_XSEQ_LEN=$(params "['max_seq_len']")

#SHOULD ADD SOMETHING HERE FOR A 2070?

# # Nvidia 2070, fp32
# PER_DEVICE_TRN_BSZ=4
# PER_DEVICE_VAL_BSZ=8
# GRAD_ACCU_STEPS=6

# # # Nvidia 2080Ti (11Gb), fp32
PER_DEVICE_TRN_BSZ=8
PER_DEVICE_VAL_BSZ=16
GRAD_ACCU_STEPS=4

# # Nvidia V100 (16Gb), fp32
# PER_DEVICE_TRN_BSZ=16
# PER_DEVICE_VAL_BSZ=32
# GRAD_ACCU_STEPS=2

#sample / testing
# MAX_STEPS=10
# WARMUP_STEPS=3
# LOGGING_STEPS=2
# LEARNING_RATE=5e-1

# #HYPERPARAMETERS for MIMIC: can change
MAX_STEPS=$(params "['xbert_model_training']['max_steps']")
WARMUP_STEPS=$(params "['xbert_model_training']['warmup_steps']")
LOGGING_STEPS=$(params "['xbert_model_training']['logging_steps']")
LEARNING_RATE=$(params "['xbert_model_training']['learning_rate']")

MODEL_DIR=${OUTPUT_DIR}/${INDEXER_NAME}/matcher/${MODEL_FOLDER_NAME}
echo python xbert/transformer.py \
    -m ${MODEL_TYPE} -n ${MODEL_NAME} \
    --do_eval -o ${MODEL_DIR} \
    -x_trn ${PROC_DATA_DIR}/X.trn.${MODEL_TYPE}.pkl \
    -c_trn ${PROC_DATA_DIR}/C.trn.${INDEXER_NAME}.npz \
    -x_tst ${PROC_DATA_DIR}/X.tst.${MODEL_TYPE}.pkl \
    -c_tst ${PROC_DATA_DIR}/C.tst.${INDEXER_NAME}.npz \
    --per_device_eval_batch_size ${PER_DEVICE_VAL_BSZ}


# predict - single GPU
CUDA_VISIBLE_DEVICES=0 python xbert/transformer.py \
    -m ${MODEL_TYPE} -n ${MODEL_NAME} \
    --do_eval -o ${MODEL_DIR} \
    -x_trn ${PROC_DATA_DIR}/X.trn.${MODEL_TYPE}.pkl \
    -c_trn ${PROC_DATA_DIR}/C.trn.${INDEXER_NAME}.npz \
    -x_tst ${PROC_DATA_DIR}/X.tst.${MODEL_TYPE}.pkl \
    -c_tst ${PROC_DATA_DIR}/C.tst.${INDEXER_NAME}.npz \
    --per_device_eval_batch_size ${PER_DEVICE_VAL_BSZ}

# # predict - multi GPU
# CUDA_VISIBLE_DEVICES=${GPID} python -u xbert/transformer.py \
#     -m ${MODEL_TYPE} -n ${MODEL_NAME} \
#     --do_eval -o ${MODEL_DIR} \
#     -x_trn ${PROC_DATA_DIR}/X.trn.${MODEL_TYPE}.pkl \
#     -c_trn ${PROC_DATA_DIR}/C.trn.${INDEXER_NAME}.npz \
#     -x_tst ${PROC_DATA_DIR}/X.tst.${MODEL_TYPE}.pkl \
#     -c_tst ${PROC_DATA_DIR}/C.tst.${INDEXER_NAME}.npz \
#     --per_device_eval_batch_size ${PER_DEVICE_VAL_BSZ}