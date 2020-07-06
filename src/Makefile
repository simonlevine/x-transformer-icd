PYTEST=pytest
PY=./.venv/bin/python
ST=streamlit
DVC=dvc
OPTS=""

.PHONY:
	@echo ""

default: app

init:  ## create python enviornment for prepare_for_xbert stage
	python -m venv .venv
	./.venv/bin/pip install -r requirements.txt

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

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
