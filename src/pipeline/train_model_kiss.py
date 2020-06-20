"""KISS Model
Build an ONNX version of the simplest possible model that still
(sort of) accomplishes our goal, mostly for rapid prototyping
and identifying pain points"""

import typing as t
IcdCode = int

import collections
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType


NOTES2DIAGNOSIS_ICD_TRAIN_FP = "./intermediary-data/notes2diagnosis-icd-train.json.gz"
MODEL_WEIGHTS_OUT_FP = "./app/assets/kiss_model.onnx"
MODEL_METADATA_FP = MODEL_WEIGHTS_OUT_FP + ".metadata.json"


def main(subsample=True):
    df_train = pd.read_json(NOTES2DIAGNOSIS_ICD_TRAIN_FP, orient="split")
    if subsample:
        df_train = df_train.sample(n=10_000)
    X_train, y_train, icd9_codes_relevant = dataframe2tensor(df_train, num_most_common=10)
    model = get_model()
    model.fit(X_train, y_train)
    model_onnx = convert_model2onnx(model, X_train)
    with open(MODEL_WEIGHTS_OUT_FP, "wb") as f_out , \
         open(MODEL_METADATA_FP, "w") as f_metadata_out:
        f_out.write(model_onnx.SerializeToString())
        json.dump({
            "model_fname": Path(MODEL_WEIGHTS_OUT_FP).name,
            "icd9_codes_relevant": icd9_codes_relevant
        }, f_metadata_out)


def dataframe2tensor(df: pd.DataFrame,
                     num_most_common: int,
                     icd_codes: t.List[IcdCode] = None
                     ) -> np.ndarray:
    """convert a dataframe, cleaned up in the previous step,
    into a tensor representing the most common codes
    
    Arguments:
        df: cleaned data, containing the ICD code and the free text only
        num_most_common: how many codes to represent
        icd_codes: the relevant codes, which also specify their ordering
    """
    if icd_codes is None:
        icd_distribution = collections.Counter(sum(df.ICD9_CODE, []))
        icd_codes, _ = list(zip(*icd_distribution.most_common(num_most_common)))
        icd_codes = list(icd_codes)
    else:
        assert len(icd_codes) == num_most_common
    for icd_code in icd_codes:
        df[icd_code] = df.ICD9_CODE.apply(lambda codes_associated_with_row: icd_code in codes_associated_with_row).astype("int64")
    df = df[df[icd_codes].sum(axis=1) > 0]  # filter out entries without codes of interest
    X = df["TEXT"].values
    y = df[icd_codes].values
    return X, y, icd_codes


def get_model():
    return Pipeline(steps=[
        ("vectorize", TfidfVectorizer()),
        ("dimensionality_reduction", TruncatedSVD(n_components=50)),
        ("classifier", RandomForestClassifier()),
    ])



def convert_model2onnx(model, training_data):
    return skl2onnx.to_onnx(
        model,
        training_data,
        "autoicd-kiss",
        initial_types=[
            ( "free_text_input", StringTensorType(shape=[None ,1]) )
        ],
        options={
            # stolen shamelessly from the ONNX docs:
            # http://onnx.ai/sklearn-onnx/auto_examples/plot_tfidfvectorizer.html
            TfidfVectorizer: {
                "separators": [
                    ' ', '.', '\\?', ',', ';', ':', '!',
                    '\\(', '\\)', '\n', '"', "'",
                    "-", "\\[", "\\]", "@"
                ]
            }
        }
    )


if __name__ == "__main__":
    main()