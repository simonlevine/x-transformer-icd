# this makefile contains the command line bindings for running the data
# pipeline and serving the user interface

PY=python
PYTEST=pytest
ST=streamlit
DVC=dvc
OPTS=""

.PHONY:
	@echo ""

default: app

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

prepare_for_xbert: ## format MIMIC data apropriately for x-BERT
	mkdir -p ./intermediary-data/xbert_inputs/ \
	&& $(DVC) run -n prepare_for_xbert \
			      -d pipeline/xbert_preprocessing.py \
			      -d '../data/mimiciii-14/DIAGNOSES_ICD.csv.gz' \
			      -d '../data/mimiciii-14/D_ICD_DIAGNOSES.csv.gz' \
			      -d '../data/ICD_general_equivalence_mapping.csv' \
			      -o './intermediary-data/xbert_inputs/label_map.txt' \
			      -o './intermediary-data/xbert_inputs/train_raw_labels.txt' \
			      -o './intermediary-data/xbert_inputs/test_raw_labels.txt' \
			      -o './intermediary-data/xbert_inputs/X.trn.npz' \
			      -o './intermediary-data/xbert_inputs/X.tst.npz' \
			      -o './intermediary-data/xbert_inputs/Y.trn.npz' \
			      -o './intermediary-data/xbert_inputs/Y.tst.npz' \
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
	export PYTHONPATH="$$PYTHONPATH:$(PWD)" && \
	$(PYTEST) -vv $(OPTS) tests/test_*

unit:
	$(MAKE) test OPTS="-m unit"
