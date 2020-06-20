# this makefile contains the command line bindings for running the data
# pipeline and serving the user interface

PY=python
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

train:  ## model training
	# inputs:
	#     "notes2diagnosis-icd-train.json.gz"
	# outputs:
	#     "./app/kiss_model.onnx"
	$(PY) pipeline/train_model_kiss.py
	
app: .PHONY  ## run streamlit app locally
	export PYTHONPATH="$(PWD)/src/app:$$PYTHONPATH" \
	&& $(ST) run app/streamlit/AutoICD.py