from pathlib import Path
import pandas as pd

DIAGNOSIS_CSV_FP = "../data/mimiciii-14/DIAGNOSES_ICD.csv.gz"
NOTE_EVENTS_CSV_FP = "../data/mimiciii-14/NOTEEVENTS.csv.gz"

def main():
    diagnosis_df = pd.read_csv(DIAGNOSIS_CSV_FP, usecols=["HADM_ID", "ICD9_CODE"])
    note_events_df = pd.read_csv(NOTE_EVENTS_CSV_FP, usecols=["HADM_ID", "TEXT"])
    df = note_events_df.merge(diagnosis_df).dropna()
    df_train = df.sample(frac=0.66, random_state=42)
    df_test = df.drop(df_train.index)

    (basedir_outpath := Path("./intermediary-data")).mkdir(exist_ok=True)
    for df_, type_ in [(df_train, "train"), (df_test, "test")]:
        fp_out = basedir_outpath/f"notes2diagnosis-icd-{type_}.json"
        print(f"Serializing {type_} dataframe to {fp_out}...")
        df_.to_json(fp_out)

if __name__ == "__main__":
    main()