import pytest
from pipeline import format_data_for_training


@pytest.fixture(scope="module")
def mimic_dataset():
    full_df, other_dfs = format_data_for_training.load_mimic_dataset()
    yield full_df, other_dfs


@pytest.fixture(scope="module")
def mimic_icd10_dataset(mimic_dataset):
    full_df, other_dfs = mimic_dataset
    full_df = format_data_for_training.convert_icd9_to_icd10(full_df)
    yield full_df, other_dfs


def test_after_ingest_columns_as_expected(mimic_dataset):
    full_data, _ = mimic_dataset
    assert {
        "HADM_ID",
        "ICD9_CODE",
        "LONG_TITLE",
        "CATEGORY",
        "SEQ_NUM",
        "TEXT"
    } == set(mimic_dataset.columns)





def test_icd9_to_icd10_has_complete_coverage(mimic_dataset):
    raise NotImplementedError


@pytest.mark.parametrize()
def test_icd9_conversion(mimic_dataset):
    raise NotImplementedError


@pytest.mark.parametrize()
def test_icd9_long_description_mapping():
    raise NotImplementedError


@pytest.mark.parametrize()
def test_icd10_long_description_mapping():
    raise NotImplementedError


@pytest.mark.parametrize()
def test_no_duplicates(mimic_dataset):
    raise NotImplementedError


def test_no_rows_are_labeled_as_error(mimic_dataset):
    """there is quirk where there is a column labeled
    ISERROR but they are all missing values. Could be
    useful. Adding test to notify if this changes, as
    it could be useful."""
    full_df, (diagnosis_df, icd9_long_description_df, note_events_df) = mimic_dataset
    assert (note_events_df.ISERROR.isna() == True).all()


def test_final_dataset_consists_of_at_least_some_examples(mimic_dataset):
    full_df, _ = mimic_dataset
    n_rows n_cols = full_df.shape
    assert 100 < n_rows


def test_final_dataset_has_no_missing_values(mimic_dataset):
    full_df, _ = mimic_dataset
    assert full_df.shape == full_df.dropna().shape


def test_train_test_validation_split_ratio_as_expected(mimic_dataset):
    raise NotImplementedError