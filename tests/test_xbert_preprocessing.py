import pytest
from pipeline import xbert_preprocessing as xp

@pytest.mark.unit
@pytest.mark.parametrize("label_raw,label_expected", [
    ("Tuberculous peritonitis",
     "Tuberculous peritonitis"),
    ("Tuberculosis of eye, unspecified",
     "Tuberculosis of eye  unspecified"),
    ("Tuberculosis of (inner) (middle) ear",
     "Tuberculosis of (inner) (middle) ear"),
    ("Chronic myeloid leukemia, BCR/ABL-positive, in relapse",
     "Chronic myeloid leukemia  BCR ABL-positive  in relapse")
])
def test_xbert_clean_label(label_raw, label_expected):
    assert xp.xbert_clean_label(label_raw) == label_expected