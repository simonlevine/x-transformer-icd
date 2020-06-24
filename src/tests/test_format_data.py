import pytest
from pipeline import format_data_for_training

## fixtures ##

@pytest.fixture(scope="module")
def mimic_dataset():
    full_df, other_dfs = format_data_for_training.load_mimic_dataset()
    yield full_df, other_dfs


@pytest.fixture(scope="module")
def gem_df():
    return format_data_for_training.load_icd_general_equivalence_mapping()


@pytest.fixture(scope="module")
def mimic_icd10_dataset(mimic_dataset, gem_df):
    full_df, other_dfs = mimic_dataset
    full_df = format_data_for_training.convert_icd9_to_icd10(full_df, gem_df)
    yield full_df, other_dfs

### fixtures ##

def test_after_ingest_columns_as_expected(mimic_dataset):
    full_data, _ = mimic_dataset
    assert {
        "HADM_ID",
        "ICD9_CODE",
        "LONG_TITLE",
        "SEQ_NUM",
        "TEXT",
    } == set(full_data.columns)


@pytest.mark.xfail(reason="we do not have full icd9-to-icd10 coverage")
def test_icd9_to_icd10_has_complete_coverage(mimic_dataset, gem_df):
    icd9_df, _ = mimic_dataset
    assert set(icd9_df.ICD9_CODE).issubset(set(gem_df.index.get_level_values("ICD9_CODE")))


@pytest.mark.xfail(reason="because we do not have full coverage, "
                          "rows with unknown ICD10 codes are dropped")
def test_icd9_to_icd10_conversion_does_not_affect_n_rows(mimic_dataset, mimic_icd10_dataset):
    icd9_df, _ = mimic_dataset
    icd10_df, _ = mimic_icd10_dataset
    n_rows_icd9, _ = icd9_df.shape
    n_rows_icd10, _ =  icd10_df.shape
    assert n_rows_icd9 == n_rows_icd10


def test_column_names_correct_after_icd9_to_icd10_conversion(mimic_icd10_dataset):
    full_icd10_df, _ = mimic_icd10_dataset
    assert {
        'HADM_ID',
        'TEXT',
        'SEQ_NUM',
        'ICD9_CODE',
        'LONG_TITLE_ICD9',
        'ICD10_CODE',
        'LONG_TITLE_ICD10'
    } == set(full_icd10_df.columns)
    assert "LONG_TITLE" not in full_icd10_df.columns  # because this would be ambiguous


@pytest.mark.parametrize("icd9,icd9_desc,icd10,icd10_desc", [
    ('01193', 'Pulmonary tuberculosis, unspecified, tubercle bacilli found (in sputum) by microscopy', 'A15.0', 'Tuberculosis of lung'),
    ('2639', 'Unspecified protein-calorie malnutrition', 'E46', 'Unspecified protein-calorie malnutrition'),
    ('2761', 'Hyposmolality and/or hyponatremia', 'E87.1', 'Hypo-osmolality and hyponatremia'),
    ('2762', 'Acidosis', 'E87.2', 'Acidosis'),
    ('2113', 'Benign neoplasm of colon', 'D12.6', 'Benign neoplasm of colon, unspecified'),
    ('4254', 'Other primary cardiomyopathies', 'I42.5', 'Other restrictive cardiomyopathy'),
    ('42731', 'Atrial fibrillation', 'I48.91', 'Unspecified atrial fibrillation'),
    ('486', 'Pneumonia, organism unspecified', 'J18.9', 'Pneumonia, unspecified organism'),
    ('5070', 'Pneumonitis due to inhalation of food or vomitus', 'J69.0', 'Pneumonitis due to inhalation of food and vomit'),
    ('5119', 'Unspecified pleural effusion', 'J91.8', 'Pleural effusion in other conditions classified elsewhere'),
    # should add an example where one icd9 is mapped to 2 icd10
])
def test_icd9_conversion(mimic_icd10_dataset, icd9, icd9_desc, icd10, icd10_desc):
    full_icd10_df, _ = mimic_icd10_dataset
    subset = full_icd10_df[full_icd10_df.ICD9_CODE == icd9]
    assert set(subset.ICD9_CODE) == {icd9,}
    assert set(sum(subset.ICD10_CODE.apply(list), [])) == {icd10,}
    assert set(subset.LONG_TITLE_ICD9) == {icd9_desc,}
    assert set(sum(subset.LONG_TITLE_ICD10.apply(list), [])) == {icd10_desc,}


@pytest.mark.parametrize("relevant_columns", [
    ("HADM_ID", "TEXT", "SEQ_NUM"),
    ("HADM_ID", "TEXT"),
    ("HADM_ID",),
])
def test_no_duplicates(mimic_icd10_dataset, relevant_columns):
    full_icd10_df, _ = mimic_icd10_dataset
    n_rows_normal, _ = full_icd10_df.shape
    n_rows_deduped, _ =  full_icd10_df.drop_duplicates(relevant_columns).shape
    assert n_rows_normal == n_rows_deduped


def test_no_rows_are_labeled_as_error(mimic_dataset):
    """there is quirk where there is a column labeled
    ISERROR but they are all missing values. Could be
    useful. Adding test to notify if this changes, as
    it could be useful."""
    full_df, (diagnosis_df, icd9_long_description_df, note_events_df) = mimic_dataset
    assert (note_events_df.ISERROR.isna() == True).all()


def test_final_dataset_consists_of_at_least_some_examples(mimic_dataset):
    full_df, _ = mimic_dataset
    n_rows, n_cols = full_df.shape
    assert 100 < n_rows


def test_final_dataset_has_no_missing_values(mimic_icd10_dataset):
    full_df, _ = mimic_icd10_dataset
    assert full_df.shape == full_df.dropna().shape


def test_train_test_validation_split_ratio_as_expected(mimic_icd10_dataset):
    tolerance = 0.5
    df, _ = mimic_icd10_dataset
    df_train, df_test = format_data_for_training.test_train_validation_split(df)
    assert (abs((2 * len(df_test)) - len(df_train))) / len(df_train) < tolerance