import typing as t

import scipy.sparse as smat
from sklearn.preprocessing import normalize as sk_normalize
from pipeline.xbert_preprocessing import
from xbert.ranker import LinearModel
from xbert.rf_linear import PostProcessor, HierarchicalMLModel


def embedding():
    #
    X1 = HierarchicalMLModel.load_feature_matrix('./data/intermediary-data/xbert_inputs/X.tst.npz')
    # xbert_trained_model_prediction.sh
    X2 = HierarchicalMLModel.load_feature_matrix('./data/intermediary-data/xbert_outputs/pifa-tfidf-s0/matcher/tst_embeddings.npy')
    X = smat.hstack([
        sk_normalize(X1, axis=1),
        sk_normalize(X2, axis=1)]
    ).tocsr()
    return X

def inference_factory(icd_9_codes:t.List[str],
                      ranker_folder,
                      transform="noop",
                      beam_size=10):

    model = LinearModel.load(ranker_folder)[-1]

    def inference(discharge_summary: str):
        # TODO: embedding step
        # Xt = embedding(discharge_summary)
        Xt = None
        # TODO: add the _pred.npz generation code here
        # tempfile.open("C_pred.npz") as f:
        #     f.write(...)
        # csr_codes = "C_pred.npz"
        csr_codes = None
        Y_pred = model.predict(
            Xt,
            csr_codes=csr_codes,
            beam_size=beam_size,
            cond_prob=PostProcessor.get(transform)
        )
        assert len(icd_9_codes) == len(Y_pred)
        return dict(zip(icd_9_codes, Y_pred))

    return inference


if __name__ == "__main__":
    icd9_codes = open('data/intermediary-data/xbert_inputs/label_map.txt').read().strip().split()
    Y_pred = inference_factory(
        icd9_codes,
        './data/intermediary-data/xbert_outputs/pifa-tfidf-s0/ranker'
    )('Lorem ipsum dolor sit amet')
    print(Y_pred)