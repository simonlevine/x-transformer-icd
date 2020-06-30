###1) BEGIN LABEL PREPROCESSING.

#!/bin/bash
DATASET = $'mimiciii-14'
LABEL_EMB=$'pifa-tfidf'

# setup label embedding feature path
#overwritten by Simon Levine for mimic.
DATA_DIR=datasets
label_emb_inst_path=${DATA_DIR}/${DATASET}/X.trn.npz

# construct label embedding
OUTPUT_DIR=save_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
mkdir -p ${PROC_DATA_DIR}

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

### 2) BEGIN FEATURE PREPROCESSING ###

#these carry through to training the transformer.
MAX_XSEQ_LEN= $'128'
#$3 #NEED TO MODIFY THIS? number tokens, = 128 by default. Perhaps too short.
MODEL_TYPE=$'bert' #keep 'bert' since this is the same model type and code relies on it.
MODEL_NAME= $'Bio_ClinicalBERT'

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


###### end#####

### 3) BEGIN TRAINING TRANSFORMER

GPID = $'0' #CUDA visible devices. 0 for just one (2070?) GPU.
    #see readme for example on >1 GPU

INDEXER_NAME=$'pifa-tfidf-s0'#as currently implemented.

#SHOULD ADD SOMETHING HERE FOR A 2070(?)

# # Nvidia 2080Ti (11Gb), fp32
PER_DEVICE_TRN_BSZ=8
PER_DEVICE_VAL_BSZ=16
GRAD_ACCU_STEPS=4

# # Nvidia V100 (16Gb), fp32
# PER_DEVICE_TRN_BSZ=16
# PER_DEVICE_VAL_BSZ=32
# GRAD_ACCU_STEPS=2

#HYPERPARAMETERS for MIMIC: can change
MAX_STEPS=1000
WARMUP_STEPS=100
LOGGING_STEPS=50
LEARNING_RATE=5e-5

MODEL_DIR=${OUTPUT_DIR}/${INDEXER_NAME}/matcher/${MODEL_NAME}
mkdir -p ${MODEL_DIR}


# train
CUDA_VISIBLE_DEVICES=${GPID} python -m torch.distributed.launch \
    --nproc_per_node 8 xbert/transformer.py \
    -m ${MODEL_TYPE} -n ${MODEL_NAME} --do_train \
    -x_trn ${PROC_DATA_DIR}/X.trn.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_trn ${PROC_DATA_DIR}/C.trn.${INDEXER_NAME}.npz \
    -o ${MODEL_DIR} --overwrite_output_dir \
    --per_device_train_batch_size ${PER_DEVICE_TRN_BSZ} \
    --gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
    --max_steps ${MAX_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --logging_steps ${LOGGING_STEPS} \
    |& tee ${MODEL_DIR}/log.txt


# predict
CUDA_VISIBLE_DEVICES=${GPID} python -u xbert/transformer.py \
    -m ${MODEL_TYPE} -n ${MODEL_NAME} \
    --do_eval -o ${MODEL_DIR} \
    -x_trn ${PROC_DATA_DIR}/X.trn.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_trn ${PROC_DATA_DIR}/C.trn.${INDEXER_NAME}.npz \
    -x_tst ${PROC_DATA_DIR}/X.tst.${MODEL_TYPE}.${MAX_XSEQ_LEN}.pkl \
    -c_tst ${PROC_DATA_DIR}/C.tst.${INDEXER_NAME}.npz \
    --per_device_eval_batch_size ${PER_DEVICE_VAL_BSZ}

#### end ####

### 3) BEGIN TRANSFORMER PREDICT STEPS####

# - (1) train linear rankers to map instances 
#       and predicted cluster scores to label scores
# - (2) output top-k predicted labels

#!/bin/bash

DATASET=$'mimiciii-14'
DATA_DIR=./datasets/${DATASET}

LABEL_NAME = pifa-tfidf-s0
EXP_NAME=${DATASET}.final

PRED_NPZ_PATHS=""
OUTPUT_DIR=save_models/${DATASET}/${LABEL_NAME}
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

### END###


##### OPTIONAL PORTIONS ####

#!/bin/bash

DATASET=$'mimiciii-14'
VERSION=$'v0'
#VERSION:
# v0=sparse TF-IDF features.
# v1=sparse TF-IDF features concatenate with dense fine-tuned XLNet embedding


LABEL_EMB=pifa-tfidf
DATA_DIR=./datasets/${DATASET}

PRED_NPZ_PATHS=""
SEED_LIST=( 0 1 2 )
for SEED in "${SEED_LIST[@]}"; do
    # indexer (for reproducibility, use clusters from pretrained_dir)
    OUTPUT_DIR=pretrained_models/${DATASET}/${LABEL_EMB}-s${SEED}
    INDEXER_DIR=${OUTPUT_DIR}/indexer
    RANKER_DIR=${OUTPUT_DIR}/ranker/linear-${VERSION}
    mkdir -p ${RANKER_DIR}

    # ranker train and predict
    PRED_NPZ_PATH=${RANKER_DIR}/tst.pred.npz

    # x_emb=TF-IDF, model=Parabel
    if [ ${VERSION} == 'v0' ]; then
        python -m xbert.ranker train \
            -x ${DATA_DIR}/X.trn.npz \
            -y ${DATA_DIR}/Y.trn.npz \
            -c ${INDEXER_DIR}/code.npz \
            -o ${RANKER_DIR} -t 0.01

        python -m xbert.ranker predict \
            -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
            -x ${DATA_DIR}/X.tst.npz \
            -y ${DATA_DIR}/Y.tst.npz

    # x_emb=xlnet_finetuned+TF-IDF, model=Parabel
    elif [ ${VERSION} == 'v1' ]; then
        python -m xbert.ranker train \
            -x ${DATA_DIR}/X.trn.npz \
            -x2 ${DATA_DIR}/X.trn.finetune.xlnet.npy \
            -y ${DATA_DIR}/Y.trn.npz \
            -c ${INDEXER_DIR}/code.npz \
            -o ${RANKER_DIR} -t 0.01 -f 0

        python -m xbert.ranker predict \
            -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
            -x ${DATA_DIR}/X.tst.npz \
            -x2 ${DATA_DIR}/X.tst.finetune.xlnet.npy \
            -y ${DATA_DIR}/Y.tst.npz -f 0

    else
        echo 'unknown linear version'
        exit
    fi

    # append all prediction path
    PRED_NPZ_PATHS="${PRED_NPZ_PATHS} ${PRED_NPZ_PATH}"
done

# final eval
EVAL_DIR=results_linear
mkdir -p ${EVAL_DIR}
python -u -m xbert.evaluator \
    -y datasets/${DATASET}/Y.tst.npz \
    -e -p ${PRED_NPZ_PATHS} \
    |& tee ${EVAL_DIR}/${DATASET}.${VERSION}.txt



### END###


# eval_transformer:
# Given the provided indexing codes (label-to-cluster assignments)
# and the fine-tuned Transformer models, train/predict ranker of 
# the X-Transformer framework, and evaluate with Precision/Recall@k:


#!/bin/bash

DATASET = $'mimiciii-14'
DATA_DIR=./datasets/${DATASET}

EXP_NAME=${DATASET}.final

LABEL_NAME = $'pifa-tfidf-s0'
MODEL_NAME= $'Bio_ClinicalBERT'

PRED_NPZ_PATHS=""

OUTPUT_DIR=pretrained_models/${DATASET}/${LABEL_NAME}
INDEXER_DIR=${OUTPUT_DIR}/indexer
MATCHER_DIR=${OUTPUT_DIR}/matcher/${MODEL_NAME}
RANKER_DIR=${OUTPUT_DIR}/ranker/${MODEL_NAME}
mkdir -p ${RANKER_DIR}

# train linear ranker
python -m xbert.ranker train \
    -x1 ${DATA_DIR}/X.trn.npz \
    -x2 ${MATCHER_DIR}/trn_embeddings.npy \
    -y datasets/${DATASET}/Y.trn.npz \
    -z ${MATCHER_DIR}/C_trn_pred.npz \
    -c ${OUTPUT_DIR}/indexer/code.npz \
    -o ${RANKER_DIR} -t 0.01 \
    -f 0 -ns 0 --mode ranker \

# predict final label ranking, using transformer's predicted cluster scores
PRED_NPZ_PATH=${RANKER_DIR}/tst.pred.npz
python -m xbert.ranker predict \
    -m ${RANKER_DIR} -o ${PRED_NPZ_PATH} \
    -x1 datasets/${DATASET}/X.tst.npz \
    -x2 ${MATCHER_DIR}/tst_embeddings.npy \
    -y datasets/${DATASET}/Y.tst.npz \
    -z ${MATCHER_DIR}/C_tst_pred.npz \
    -f 0 -t noop

# append all prediction path
PRED_NPZ_PATHS="${PRED_NPZ_PATHS} ${PRED_NPZ_PATH}"
# done

# final eval
EVAL_DIR=results_transformer
mkdir -p ${EVAL_DIR}
python -u -m xbert.evaluator \
    -y datasets/${DATASET}/Y.tst.npz \
    -e -p ${PRED_NPZ_PATHS} |& tee ${EVAL_DIR}/${EXP_NAME}.txt




# Evaluate Linear Models
# Given the provided indexing codes (label-to-cluster assignments), train/predict linear models, and evaluate with Precision/Recall@k:


### END
