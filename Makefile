CONDA_ENV_NAME=`grep name: environment.yml | sed -e 's/name: //' | cut -d "'" -f 2 | cut -d '"' -f 2`
OPTS=""

default: app

.PHONY:
	@echo ""
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
init:  ## set up development enviornment
	python -m venv .venv \
	&& ./.venv/bin/pip install -r requirements-app.txt

	pipx install "dvc[all]"

	conda env create -f environment.yml \
	&& conda activate $(CONDA_ENV_NAME) \
	&& pip install -e . \
    && pip install appdirs attrs black boto3 \
                   botocore chardet click docutils \
                   idna jmespath pathspec python-dateutil \
                   regex requests s3transfer sacremoses \
                   sentencepiece toml torch tqdm transformers \
                   typed-ast urllib3 loguru \
    && python setup.py install --force
uninstall:  ## tear down development enviornment
	eval "$$(conda shell.bash hook)" \
	&& conda deactivate \
	&& conda remove -y -n $(CONDA_ENV_NAME) --all
	rm -rf .venv
train:  ## model training
	dvc repro
app: .PHONY  ## run streamlit app locally
	$(ST) run app/streamlit/AutoICD.py
test:  # pytest suite
	export PYTHONPATH="$$PYTHONPATH:$(PWD)" && \
	$(PYTEST) -vv $(OPTS) tests/test_*
unit:  ## pytest suite, unit tests only 
	$(MAKE) test OPTS="-m unit"