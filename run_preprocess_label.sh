#!/bin/bash

# DATASET=$1
DATASET = $'mimiciii-14'

LABEL_EMB=$'pifa-tfidf'


# setup label embedding feature path
# overwrite it if necessary
DATA_DIR=datasets
label_emb_inst_path=${DATA_DIR}/${DATASET}/X.trn.npz


# construct label embedding
OUTPUT_DIR=save_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
mkdir -p ${PROC_DATA_DIR}
: "
python -u -m xbert.preprocess \
	--do_label_embedding \
	-i ${DATA_DIR}/${DATASET} \
    -o ${PROC_DATA_DIR} \
    -l ${LABEL_EMB} \
    -x ${label_emb_inst_path}


# semantic label indexing
SEED_LIST=( 0 1 2 )
for SEED in "${SEED_LIST[@]}"; do
    LABEL_EMB_NAME=${LABEL_EMB}-s${SEED}
	INDEXER_DIR=${OUTPUT_DIR}/${LABEL_EMB_NAME}/indexer
	python -u -m xbert.indexer \
		-i ${PROC_DATA_DIR}/L.${LABEL_EMB}.npz \
		-o ${INDEXER_DIR} --seed ${SEED}
done
"

# construct C.[trn|tst].[label-emb].npz for training matcher
SEED=0
LABEL_EMB_NAME=${LABEL_EMB}-s${SEED}
INDEXER_DIR=${OUTPUT_DIR}/${LABEL_EMB_NAME}/indexer
python -u -m xbert.preprocess \
    --do_proc_label \
    -i ${DATA_DIR}/${DATASET} \
    -o ${PROC_DATA_DIR} \
    -l ${LABEL_EMB_NAME} \
    -c ${INDEXER_DIR}/code.npz

#### end ####

