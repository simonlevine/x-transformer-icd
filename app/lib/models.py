"""deserialize auto-icd models and provide a consistent interface"""

import typing as t
import json
import pickle
from pathlib import Path
import numpy as np
import onnxruntime as rt

APP_ROOT = Path("./app")
ASSETS_DIR = APP_ROOT/"assets"


class AutoICDModel:

    def __init__(self, onnx_model_fp):
        assert onnx_model_fp.exists()
        self.sess = rt.InferenceSession(str(onnx_model_fp.resolve()))

    def ___call__(self, free_text: str) -> t.Set[str]:
        raise NotImplementedError("Subclasses just provide model interaction logic!")


# class KissModel(AutoICDModel):

#     def __init__(self, onnx_model_fp, icd9_codes: t.List[str]):
#         """because we are only loading a few codes,
#         we need to know which ones in otder to decode
#         decode the model output, which is a 1x|icd9_codes| matrix"""
#         super().__init__(onnx_model_fp)
#         self.icd9_codes = icd9_codes

#     def ___call__(self, free_text: str) -> t.Set[str]:
#         X = np.array([[free_text]])
#         predictions, predictions_proba \
#             = sess.run(None, {"free_text_input": X})[0]
#         codes_predicted = [
#             code for prediction, code in zip(predictions, self.icd9_codes)
#             if prediction == 1  # i.e., if the code is predicted to be present
#         ]
#         codes2predicted_proba = {
#             code: proba for code, proba in zip(self.icd9_codes, predictions_proba)
#         }
#         return codes_predicted, codes2predicted_proba


# def get_kiss_model():
#     onnx_model_fp = ASSETS_DIR/"kiss_model.onnx"
#     with open(ASSETS_DIR/"kiss_model.onnx.metadata.json") as f:
#         icd9_codes = json.load(f)["icd9_codes_relevant"]
#     model = KissModel(onnx_model_fp, icd9_codes)
#     return model


class KissModel:
    """Kiss Model using pickle for persistence"""

    def __init__(self):
        with open(ASSETS_DIR/"kiss_model.pkl.metadata.json") as f_meta:
            self.icd9_codes = json.load(f_meta)["icd9_codes_relevant"]
        with open(ASSETS_DIR/"kiss_model.pkl", "rb") as f:
            self.model = pickle.loads(f.read())
    
    def __call__(self, free_text: str):
        X = np.array([free_text])
        predicted_codes_proba = self.model.predict_proba(X)
        return np.array([proba.tolist() for proba in predicted_codes_proba])[:,0,1]