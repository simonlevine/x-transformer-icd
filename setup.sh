#!/bin/bash

cd auto-icd-transformers
conda env create -f environment.yml
source activate pt1.2_xmlc_transformer
pip install -e .
python setup.py install --force
