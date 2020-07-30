import numpy as np
import pandas as pd
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import string
import re
import itertools
import pickle
import warnings


import spacy
import stanfordnlp
import time
import scispacy
from tqdm import tqdm
import itertools
import pickle
import sys
import re
import nltk
import os
from heuristic_tokenize import sent_tokenize_rules


DIAGNOSIS_CSV_FP = "./data/mimiciii-14/DIAGNOSES_ICD.csv.gz"
PROCEDURE_CSV_FP = "./data/mimiciii-14/PROCEDURES_ICD.csv"
NOTE_EVENTS_CSV_FP = "./data/mimiciii-14/NOTEEVENTS.csv.gz"
ICD9_DIAG_KEY_FP = "./data/mimiciii-14/D_ICD_DIAGNOSES.csv.gz"
ICD9_PROC_KEY_FP = "./data/mimiciii-14/D_ICD_PROCEDURES.csv.gz"
ICD_GEM_FP = "./data/ICD_general_equivalence_mapping.csv"


def generate_notes_df():
    note_event_cols = ["HADM_ID", "CATEGORY", "TEXT"]
    note_events_df = pd.read_csv(NOTE_EVENTS_CSV_FP, usecols=note_event_cols)
    note_events_df = note_events_df.drop_duplicates(["TEXT"]).groupby(
        ['HADM_ID']).agg({'TEXT': ' '.join, 'CATEGORY': ' '.join})
    return note_events_df

def load_diag_and_procs():
    diagnoses_icd = pd.read_csv(
        DIAGNOSIS_CSV_FP, compression='gzip')
    procedures_icd = pd.read_csv(
        PROCEDURE_CSV_FP, compression='gzip')

    return diagnoses_icd, procedures_icd

def generate_dicts(diagnoses_icd, procedures_icd):
    diagnoses_dict = {}
    for i in range(len(diagnoses_icd)):
        entry = diagnoses_icd.iloc[i]
        hadm = entry['HADM_ID']
        icd = entry['ICD9_CODE']
        if hadm not in diagnoses_dict:
            diagnoses_dict[hadm] = [icd]
        else:
            diagnoses_dict[hadm].append(icd)

    procedures_dict = {}
    for i in range(len(procedures_icd)):
        entry = procedures_icd.iloc[i]
        hadm = entry['HADM_ID']
        icd = entry['ICD9_CODE']
        if hadm not in procedures_dict:
            procedures_dict[hadm] = [icd]
        else:
            procedures_dict[hadm].append(icd)

    return diagnoses_dict,procedures_dict


def generate_outcomes_dfs(diagnoses_dict, procedures_dict):
    diagnoses_df = pd.DataFrame.from_dict(diagnoses_dict, orient='index')
    diagnoses_df.columns = ['DIAG_CODE' + str(i) for i in range(1, len(diagnoses_df.columns)+1)]
    diagnoses_df.index.name = 'HADM_ID'
    diagnoses_df['DIAG_CODES'] = diagnoses_df[diagnoses_df.columns[:]].apply(
        lambda x: ','.join(x.dropna().astype(str)),
        axis=1
    )


    procedures_df = pd.DataFrame.from_dict(procedures_dict, orient='index')
    procedures_df.columns = ['PRCD_CODE' + str(i) for i in range(1, len(procedures_df.columns)+1)]
    procedures_df.index.name = 'HADM_ID'
    procedures_df['PROC_CODES'] = procedures_df[procedures_df.columns[:]].apply(
        lambda x: ','.join(x.dropna().astype(str)),
        axis=1
    )


    codes_df = pd.merge(diagnoses_df, procedures_df, how='outer', on='HADM_ID')

    return diagnoses_df, procedures_df, codes_df


def generate_merged_df(notes_df, diagnoses_df, procedures_df, codes_df):
    diagnoses = diagnoses_df[['DIAG_CODES']]
    procedures = procedures_df[['PROC_CODES']]
    codes = pd.merge(diagnoses, procedures, how='outer', on='HADM_ID')
    codes = codes.dropna()

    merged_df = pd.merge(notes_df, codes, how='left', on='HADM_ID')
    # merged_df = merged_df.dropna()
    
    return merged_df


def preprocess_and_clean_note(note):
    note = " ".join(note.split())
    note = remove_stopwords(note)  # remove stopwords
    note = " ".join(note)
    note = note.lower()  # make lowercase
    note = note.replace(r"\[.*?\]", "")  # remove de-id token
    note = note.replace('\n', ' ')
    note = note.replace('w/', 'with')
    note = note.replace("_", "")
    note = note.replace("#", "")
    note = re.sub(r'\d+', '', note)  # remove numbers
    note = note.translate(str.maketrans(
        '', '', string.punctuation))  # remove punctuation
    return note


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))

    other_words = {'', 'Admission Date', 'Discharge Date', 'Date of Birth', 'Phone', 'Date/Time', 'ID',
                   'Completed by', 'Dictated By', 'Attending', 'Provider: ', 'Provider', 'Primary', 'Secondary',
                   ' MD Phone', ' M.D. Phone', ' MD', ' PHD',
                   ' X', ' IV', ' VI', 'III', 'II', 'VIII',
                   'JOB#', 'JOB#: cc', '# Code',
                   'Metoprolol Tartrate 25 mg Tablet Sig', ')', '000 unit/mL Suspension Sig', ' ', '0.5 % Drops ', '   Status: Inpatient DOB', 'Levothyroxine 50 mcg Tablet Sig', '0.5 % Drops Sig', 'Lidocaine 5 %(700 mg/patch) Adhesive Patch', 'Clopidogrel Bisulfate 75 mg Tablet Sig', 'Levofloxacin 500 mg Tablet Sig', 'Albuterol 90 mcg/Actuation Aerosol ', 'None Tech Quality: Adequate Tape #', '000 unit/mL Solution Sig', 'x'
                   }

    word_tokens = word_tokenize(text)
    filtered_text = [
        word for word in word_tokens if word not in stop_words.union(other_words)]
    return filtered_text

    
#filtering steps
merged_df['TEXT'] = merged_df['TEXT'].apply(
    preprocess_and_clean_note)
