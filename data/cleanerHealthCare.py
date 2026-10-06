import os

import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(base_dir, "raw", "Healthcare.csv"))

df.drop(columns=["Age", "Symptom_Count", "Gender"], inplace=True)
df.drop_duplicates(inplace=True)

symptom_dummies = df['Symptoms'].str.get_dummies(sep=',')
symptom_dummies.columns = symptom_dummies.columns.str.strip()
symptom_dummies = symptom_dummies.T.groupby(level=0).max().T

df_updated = pd.concat([df.drop(columns=['Symptoms']), symptom_dummies], axis=1)
df_updated.to_csv(os.path.join(base_dir, "cleaned", "Healthcare_Processed.csv"), index=False)

print(f"Total unique symptom columns created: {symptom_dummies.shape[1]}")
print(df_updated.head())