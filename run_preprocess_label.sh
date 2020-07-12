#!/usr/bin/env bash
# This should yield L.${LABEL_EMB}.npz in the PROC_DATA_DIR.

DATASET=$'mimiciii-14'
LABEL_EMB=$'pifa-tfidf'

source create_conda_env_as_necessary.sh

# setup label embedding feature path
#overwritten by Simon Levine for mimic.



# construct label embedding
DATA_DIR=../../data/intermediary-data
OUTPUT_DIR=${DATA_DIR}/xbert_outputs

label_emb_inst_path=${DATA_DIR}/xbert_inputs/${DATASET}/X.trn.npz

PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
mkdir -p ${PROC_DATA_DIR}

python -m xbert.preprocess \
	--do_label_embedding \
	-i ${DATA_DIR}/xbert_inputs/${DATASET} \
    -o ${PROC_DATA_DIR} \
    -l ${LABEL_EMB} \
    -x ${label_emb_inst_path}

# semantic label indexing
# This should yield code.npz in the INDEXER_DIR

SEED_LIST=( 0 1 2 )
for SEED in "${SEED_LIST[@]}"; do
    LABEL_EMB_NAME=${LABEL_EMB}-s${SEED}
	INDEXER_DIR=${OUTPUT_DIR}/${LABEL_EMB_NAME}/indexer
	python -u -m xbert.indexer \
		-i ${PROC_DATA_DIR}/L.${LABEL_EMB}.npz \
		-o ${INDEXER_DIR} --seed ${SEED}
done

# construct C.[trn|tst].[label-emb].npz for training matcher
# This should yield the instance-to-cluster matrix C.trn.npz and C.tst.npz in the PROC_DATA_DIR.
SEED=0
LABEL_EMB_NAME=${LABEL_EMB}-s${SEED}
INDEXER_DIR=${OUTPUT_DIR}/${LABEL_EMB_NAME}/indexer
python -u -m xbert.preprocess \
    --do_proc_label \
    -i ${DATA_DIR}/xbert_inputs/${DATASET} \
    -o ${PROC_DATA_DIR} \
    -l ${LABEL_EMB_NAME} \
    -c ${INDEXER_DIR}/code.npz

#### end ####

