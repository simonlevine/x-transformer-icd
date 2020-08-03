#!/usr/bin/env bash

source params.sh

CUDA_VISIBLE_DEVICES=0
NVIDIA_VISIBLE_DEVICES=0

GPID=${0} #CUDA visible devices. 0 for just one 2070 GPU.
# NPROC_PER_NODE='1'

DATASET=$'mimiciii-14'
INDEXER_NAME=$'pifa-tfidf-s0' # or ||| pifa-neural-s0 ||| text-emb-s0

MODEL_NAME=$(params "['model_name']")
MODEL_FOLDER_NAME='roberta'
MODEL_TYPE=$'roberta'

OUTPUT_DIR=./data/intermediary-data/xbert_outputs
PROC_DATA_DIR=./data/intermediary-data/xbert_outputs/proc_data

MAX_XSEQ_LEN=$(params "['max_seq_len']")

#SHOULD ADD SOMETHING HERE FOR A 2070?

# stupid-tier
# PER_DEVICE_TRN_BSZ=1
# PER_DEVICE_VAL_BSZ=2
# GRAD_ACCU_STEPS=8

# # # Nvidia 2080Ti (11Gb), fp32
# PER_DEVICE_TRN_BSZ=8
# PER_DEVICE_VAL_BSZ=8
# GRAD_ACCU_STEPS=4

# # # Nvidia V100 (16Gb), fp32
# PER_DEVICE_TRN_BSZ=16 
# PER_DEVICE_VAL_BSZ=16 
# GRAD_ACCU_STEPS=2 

PER_DEVICE_TRN_BSZ=$(params "['xbert_model_training']['per_device_training_batchsize']")
PER_DEVICE_VAL_BSZ=$(params "['xbert_model_training']['per_device_validation_batchsize']")
GRAD_ACCU_STEPS=$(params "['xbert_model_training']['grad_accu_steps']")

MAX_STEPS=$(params "['xbert_model_training']['max_steps']")
WARMUP_STEPS=$(params "['xbert_model_training']['warmup_steps']")
LOGGING_STEPS=$(params "['xbert_model_training']['logging_steps']")
LEARNING_RATE=$(params "['xbert_model_training']['learning_rate']")


MODEL_DIR=${OUTPUT_DIR}/${INDEXER_NAME}/matcher
mkdir -p ${MODEL_DIR}

CUDA_VISIBLE_DEVICES=0 $PY_CONDA xbert/transformer.py \
    -m ${MODEL_TYPE} \
    -n ${MODEL_NAME} \
    --do_train \
    -x_trn ${PROC_DATA_DIR}/X.trn.tomodel.pkl \
    -c_trn ${PROC_DATA_DIR}/C.trn.${INDEXER_NAME}.npz \
    -o ${MODEL_DIR} --overwrite_output_dir \
    --per_device_train_batch_size ${PER_DEVICE_TRN_BSZ} \
    --gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
    --max_steps ${MAX_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --overwrite_output_dir #\
    # --fp16
    # --logging_steps ${LOGGING_STEPS}  |& tee ${MODEL_DIR}/log.txt


# #train - multi-gpu
# CUDA_VISIBLE_DEVICES=0 $PY_CONDA -m torch.distributed.launch \
#     --nproc_per_node 1 xbert/transformer.py \
#     -m ${MODEL_TYPE} -n ${MODEL_NAME} --do_train \
#     -x_trn ${PROC_DATA_DIR}/X.trn.tomodel.pkl \
#     -c_trn ${PROC_DATA_DIR}/C.trn.${INDEXER_NAME}.npz \
#     -o ${MODEL_DIR} --overwrite_output_dir \
#     --per_device_train_batch_size ${PER_DEVICE_TRN_BSZ} \
#     --gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
#     --max_steps ${MAX_STEPS} \
#     --warmup_steps ${WARMUP_STEPS} \
#     --learning_rate ${LEARNING_RATE} \
#     --logging_steps ${LOGGING_STEPS} \
#     |& tee ${MODEL_DIR}/log.txt


#### end ####

