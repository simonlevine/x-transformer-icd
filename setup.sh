#!/usr/bin/env bash

eval "$(conda shell.bash hook)"

python -m venv .venv \
&& ./.venv/bin/pip install -r requirements-app.txt

pipx install "dvc[all]"

conda env create -f environment.yml \
&& conda activate $CONDA_ENV_NAME \
pip install -e . \
&& pip install appdirs attrs black boto3 \
                botocore chardet click docutils \
                idna jmespath pathspec python-dateutil \
                regex requests s3transfer sacremoses \
                sentencepiece toml torch tqdm transformers \
                typed-ast urllib3 loguru \
&& python setup.py install --force
&& conda install pytorch torchvision cudatoolkit=10.1 -c pytorch 