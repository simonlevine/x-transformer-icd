import typing as t
import re
import itertools as it
import string
from pathlib import Path
import pandas as pd
from loguru import logger
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')		
nltk.download('stopwords')

MIMIC_DIR = Path("data/mimiciii-14/")
NOTE_EVENTS_CSV_FP = MIMIC_DIR / "NOTEEVENTS.csv.gz"
OUTDIR = Path("./data/intermediary-data/filtered_notes/")
(OUTDIR).mkdir(exist_ok=True)
FILTERED_NOTE_EVENTS_CSV_FP = OUTDIR / "NOTEEVENTS.FILTERED.csv.gz"

ADMIN_LANGUAGE = [
    "Admission Date",
    "Discharge Date",
    "Date of Birth",
    "Phone",
    "Date/Time",
    "ID",
    "Completed by",
    "Dictated By",
    "Attending",
    "Provider: ",
    "Provider",
    "Primary",
    "Secondary",
    " MD Phone",
    " M.D. Phone",
    " MD",
    " PHD",
    " X",
    " IV",
    " VI",
    " III",
    " II",
    " VIII",
    "JOB#",
    "JOB#: cc",
    "# Code",
    "Metoprolol Tartrate 25 mg Tablet Sig",
    ")",
    "000 unit/mL Suspension Sig",
    "0.5 % Drops ",
    "   Status: Inpatient DOB",
    "Levothyroxine 50 mcg Tablet Sig",
    "0.5 % Drops Sig",
    "Lidocaine 5 %(700 mg/patch) Adhesive Patch",
    "Clopidogrel Bisulfate 75 mg Tablet Sig",
    "Levofloxacin 500 mg Tablet Sig",
    "Albuterol 90 mcg/Actuation Aerosol ",
    "None Tech Quality: Adequate Tape #",
    "000 unit/mL Solution Sig",
    "x",
]
STOP_WORDS = set(stopwords.words("english"))


def main():
    logger.info(f"loading {NOTE_EVENTS_CSV_FP.name} into memory")
    notes_df = pd.read_csv(NOTE_EVENTS_CSV_FP, low_memory=False)
    notes_filtered_df = preprocess_and_clean_notes(notes_df)
    notes_filtered_df.to_csv(FILTERED_NOTE_EVENTS_CSV_FP)


def preprocess_and_clean_notes(notes_df: pd.DataFrame) -> pd.DataFrame:
    """remove redundant information from the free text, which are discharge summaries,
    using both common NLP techniques and heuristic rules

    Args:
        notes_df (pd.DataFrame): MimicIII's NOTEEVENTS.csv.gz, including the columns:
            ['ROW_ID', 'SUBJECT_ID', 'HADM_ID', 'CHARTDATE', 'CHARTTIME',
            'STORETIME', 'CATEGORY', 'DESCRIPTION', 'CGID', 'ISERROR', 'TEXT']

    Returns:
        pd.DataFrame: notes_df, filtered of redundant text
    """
    notes_df["TEXT"] = notes_df["TEXT"].str.lower()
    logger.info("Removing de-id token, punctuation, admin language and other cruft")

    with tqdm(total=2+len(ADMIN_LANGUAGE)+5) as pbar:
        for regex in [r"\[.*?\]", "[{}]".format(string.punctuation)]:
            notes_df["TEXT"] = notes_df["TEXT"].replace(regex, "", regex=True)
            pbar.update(1)
        for admin_token in ADMIN_LANGUAGE:
            notes_df["TEXT"] = notes_df["TEXT"].str.replace(admin_token, "")
            pbar.update(1)
        for original, replacement in [
            ("\n", " "),
            ("w/", "with"),
            ("_", ""),
            ("#", ""),
            ("\d+", ""),
        ]:
            notes_df["TEXT"] = notes_df["TEXT"].str.replace(original, replacement)
            pbar.update(1)
    logger.info("Removing Stopwords")
    tqdm.pandas()
    notes_df["TEXT"] = notes_df["TEXT"].progress_apply(remove_stopwords)
    
    return notes_df


def remove_stopwords(text_: str) -> str:
    """splice out stop words such as 'she', 'no', 'that', 'did', 'who', 'during', "shouldn't" or 'more'

    >>> remove_stopwords("When we are born, we cry that we are come to this great stage of fools.")
    >>> "When born , cry come great stage fools ."

    Args:
        text_ (str): a single discharge summary example

    Returns:
        str: discharge summary, more compactly represented
    """
    text_ = " ".join(text_.split())
    word_tokens = word_tokenize(text_)
    filtered_text = " ".join((word for word in word_tokens if word not in STOP_WORDS))
    return filtered_text


if __name__ == "__main__":
    main()
