#!/bin/bash
git clone --recursive -b training-on-sample https://simonlevine:viphuK-tybfiv-4guxqa@github.com/jeremyadamsfisher/auto-icd.git
git submodule update --recursive --remote
pip install dvc[gcp]
cd auto-icd
export GOOGLE_APPLICATION_CREDENTIALS=$"autoicd-gcp-credentials.json"


cd auto-icd-transformers
conda env create -f environment.yml
conda activate pt1.2_xmlc_transformer
pip install -e . loguru
python setup.py install --force
pip install -U transformers

cd .. && make init

NVIDIA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0