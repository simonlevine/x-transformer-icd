#!/bin/bash
git clone -b preprocessing_fixes_for_colab https://simonlevine:viphuK-tybfiv-4guxqa@github.com/jeremyadamsfisher/auto-icd.git
git clone -b colab-fixes https://simonlevine:viphuK-tybfiv-4guxqa@github.com/jeremyadamsfisher/auto-icd-transformers.git

pip install dvc[gcp]
cd auto-icd
export GOOGLE_APPLICATION_CREDENTIALS=$"autoicd-gcp-credentials.json"


cd auto-icd-transformers
conda env create -f environment.yml
conda activate pt1.2_xmlc_transformer
pip install -e . loguru
python setup.py install --force


