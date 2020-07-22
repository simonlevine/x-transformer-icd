OPTS=""

default: app

.PHONY:
	@echo ""
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
init:  ## set up development enviornment
	./setup.sh
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