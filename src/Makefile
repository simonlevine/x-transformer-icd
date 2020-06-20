# this makefile contains the command line bindings for running the data
# pipeline and serving the user interface

PY=python
ST=streamlit

.PHONY:
	@echo ""

default: app

format_data:
	## inputs:
	##     "../data/mimiciii-14/DIAGNOSES_ICD.csv.gz"
	##     "../data/mimiciii-14/NOTEEVENTS.csv.gz"
	## outputs:
	##     "notes2diagnosis-icd-{train,test}.json.gz"
	$(PY) pipeline/format_data_for_training.py

train:
	## inputs:
	#      "notes2diagnosis-icd-train.json.gz"
	## outputs:
	##     "./app/kiss_model.onnx"
	$(PY) pipeline/train_model_kiss.py
	
app: .PHONY
	export PYTHONPATH="$(PWD)/src/app:$$PYTHONPATH" \
	&& $(ST) run app/streamlit/AutoICD.py