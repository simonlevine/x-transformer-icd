if [[ $(conda info -e) == *pt1.2_xmlc_transformer* ]]; then
    conda env create -f environment.yml
    source activate pt1.2_xmlc_transformer
    pip install -e .
    pip install appdirs attrs black boto3 \
                botocore chardet click docutils \
                idna jmespath pathspec python-dateutil \
                regex requests s3transfer sacremoses \
                sentencepiece toml torch tqdm transformers \
                typed-ast urllib3
    python setup.py install --force
else
    source activate pt1.2_xmlc_transformer
fi