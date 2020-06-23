# this makefile contains the command line bindings for running the data
# pipeline and serving the user interface

PY=python
PYTEST=pytest
ST=streamlit
.PHONY:
	@echo ""
default: app

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

format_data:  ## data preprocessing step
	# inputs:
	#     "../data/mimiciii-14/DIAGNOSES_ICD.csv.gz"
	#     "../data/mimiciii-14/NOTEEVENTS.csv.gz"
	# outputs:
	#     "notes2diagnosis-icd-{train,test}.json.gz"
	$(PY) pipeline/format_data_for_training.py

prepare_for_xbert: ## xbertpreprocessing step.
	# inputs:
	# DIAGNOSIS_CSV_FP = "../data/mimiciii-14/DIAGNOSES_ICD.csv.gz"
	# ICD9_KEY_FP = "../data/mimiciii-14/D_ICD_DIAGNOSES.csv.gz"
	# ICD_GEM_FP = "../data/ICD_general_equivalence_mapping.csv"

	# "../src/pipeline/format_data_for_training.py"

	#outputs
		
	# XBERT_LABEL_MAP_FP = '../data/xbert_inputs/label_map.txt'

	# XBERT_TRAIN_RAW_LABELS_FP = '../data/xbert_inputs/train_raw_labels.txt'
	# XBERT_TEST_RAW_LABELS_FP = '../data/xbert_inputs/test_raw_labels.txt'

	# XBERT_X_TRN_FP = '../data/xbert_inputs/X.trn.npz'
	# XBERT_X_TST_FP = '../data/xbert_inputs/X.tst.npz'

	# XBERT_Y_TRN_FP = '../data/xbert_inputs/Y.trn.npz'
	# XBERT_Y_TST_FP = '../data/xbert_inputs/Y.tst.npz'

	$(PY) pipeline/xbert_preprocessing.py

train:  ## model training
	# inputs:
	#     "notes2diagnosis-icd-train.json.gz"
	# outputs:
	#     "./app/kiss_model.onnx"
	$(PY) pipeline/train_model_kiss.py
	
app: .PHONY  ## run streamlit app locally
	$(ST) run app/streamlit/AutoICD.py

test:
	$(PYTEST) tests/test_*