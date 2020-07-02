# create the x-transformer conda enviornment if it does not
# exist, then (in either case) activate it

# usage: % source create_conda_env_as_necessary.sh && python script.py

eval "$(conda shell.bash hook)"

if [[ $(conda info -e) == *pt1.2_xmlc_transformer* ]]; then
    conda activate pt1.2_xmlc_transformer
else
    conda env create -f environment.yml
    conda activate pt1.2_xmlc_transformer
    pip install -e .
    pip install appdirs attrs black boto3 \
                botocore chardet click docutils \
                idna jmespath pathspec python-dateutil \
                regex requests s3transfer sacremoses \
                sentencepiece toml torch tqdm transformers \
                typed-ast urllib3 loguru
    python setup.py install --force
fi

python -c 'from loguru import logger; logger.info("Activated pt1.2_xmlc_transformer enviornment!")'