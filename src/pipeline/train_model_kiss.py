"""KISS Model
Build an ONNX version of the simplest possible model that still
(sort of) accomplishes our goal, mostly for rapid prototyping
and identifying pain points"""

import typing as t
IcdCode = int

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

NOTES2DIAGNOSIS_ICD_TRAIN_FP = "./intermediary-data/notes2diagnosis-icd-train.json"
NOTES2DIAGNOSIS_ICD_TEST_FP = "./intermediary-data/notes2diagnosis-icd-test.json"
MODEL_WEIGHTS_OUT_FP = "./app/kiss_model.onnx"

def dataframe2tensor(df: pd.DataFrame,
                     num_most_common:int,
                     icd_codes:t.List[IcdCode]=None
                     ) -> np.ndarray:
    """convert a dataframe, cleaned up in the previous step,
    into a tensor representing the most common codes
    
    Arguments:
        df: cleaned data, containing the ICD code and the free text only
        num_most_common: how many codes to represent
        icd_codes: the relevant codes, which also specify their ordering
    """
    if icd_codes is None:
        icd9_codes = list(df.ICD9_CODE.unique())
    ddf = pd.DataFrame(index=df.HADM_ID.unique(), columns=["TEXT"] + icd9_codes).fillna(False)
    for idx, row in df.iterrows():
        ddf.loc[row.HADM_ID, "TEXT"] = row.TEXT
        ddf.loc[row.HADM_ID, row.ICD9_CODE] = True
    X = ddf[["TEXT"]].values
    y = ddf[icd9_codes[0]].values.astype("int64")
    return X, y, icd_codes


def main(subsample=True):
    df_train = pd.read_json(NOTES2DIAGNOSIS_ICD_TRAIN_FP)
    # df_test = pd.read_json(NOTES2DIAGNOSIS_ICD_TEST_FP)
    if subsample:
        icd_codes_most_frequent = set(df_train["ICD9_CODE"].value_counts().index[:10])
        df_train = df_train[df_train.ICD9_CODE.isin(most_frequent)].sample(n=10_000)
    model = Pipeline(steps=[
        ("reshape", ColumnTransformer([
            ("vectorize", TfidfVectorizer(), 0),
        ])),
        ("dimensionality_reduction", TruncatedSVD(n_components=50)),
        ("classifier", RandomForestClassifier()),
    ])
    X_train, y_train, icd9_codes_relevant = dataframe2tensor(df_train, 10)
    # X_test, y_test, _ = dataframe2tensor(df_test, 10, icd9_codes_relevant)
    model_onx = convert_sklearn(
        pipeline,
        "autoicd-kiss",
        initial_types=[("free_text_input", StringTensorType(shape=[None, 1]))],
        options={
            # stolen shamelessly from the ONNX docs: http://onnx.ai/sklearn-onnx/auto_examples/plot_tfidfvectorizer.html
            TfidfVectorizer: {
                "separators": [
                    ' ', '.', '\\?', ',', ';', ':', '!',
                    '\\(', '\\)', '\n', '"', "'",
                    "-", "\\[", "\\]", "@"
                ]
            }
        }
    )
    with open(MODEL_WEIGHTS_OUT_FP, "wb") as f_out:
        f_out.write(model_onx.SerializeToString())

if __name__ == "__main__":
    main()