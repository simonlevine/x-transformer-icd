#!/usr/bin/env bash
source params.sh

DATASET=$'mimiciii-14'
LABEL_EMB=$'pifa-tfidf'
MODEL_NAME=$(params "['model_name']")
MODEL_TYPE=$'roberta'

# setup label embedding feature path
#overwritten by Simon Levine for mimic.

MAX_XSEQ_LEN=$(params "['max_seq_len']")
# construct label embedding
DATA_DIR=./data/intermediary-data
OUTPUT_DIR=${DATA_DIR}/xbert_outputs

label_emb_inst_path=${DATA_DIR}/xbert_inputs/X.trn.npz

PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
mkdir -p ${PROC_DATA_DIR}

$PY_CONDA xbert/preprocess.py \
	--do_label_embedding \
	-i ${DATA_DIR}/xbert_inputs \
    -o ${PROC_DATA_DIR} \
    -l ${LABEL_EMB} \
    -x ${label_emb_inst_path} \
    --max_xseq_len ${MAX_XSEQ_LEN} \
    --model_name_or_path ${MODEL_NAME} \
    -m ${MODEL_TYPE}


# semantic label indexing
SEED_LIST=( 0 1 2 )
for SEED in "${SEED_LIST[@]}"; do
    LABEL_EMB_NAME=${LABEL_EMB}-s${SEED}
	INDEXER_DIR=${OUTPUT_DIR}/${LABEL_EMB_NAME}/indexer
	$PY_CONDA -u -m xbert.indexer \
		-i ${PROC_DATA_DIR}/L.${LABEL_EMB}.npz \
		-o ${INDEXER_DIR} --seed ${SEED}
done

# construct C.[trn|tst].[label-emb].npz for training matcher
SEED=0
LABEL_EMB_NAME=${LABEL_EMB}-s${SEED}
INDEXER_DIR=${OUTPUT_DIR}/${LABEL_EMB_NAME}/indexer
$PY_CONDA xbert/preprocess.py \
    --do_proc_label \
    -i ${DATA_DIR}/xbert_inputs \
    -o ${PROC_DATA_DIR} \
    -l ${LABEL_EMB_NAME} \
    -c ${INDEXER_DIR}/code.npz \
    --max_xseq_len ${MAX_XSEQ_LEN}
#### end ####

