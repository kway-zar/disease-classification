
import os

import pandas as pd
import numpy as np

base_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(base_dir, "raw", "Disease_symptom_and_patient_profile_dataset.csv"))

print(df.head())
print(df.info())
print(df.isnull().sum())

df.reset_index(drop=True, inplace=True)
df.dropna(inplace=True)

print("\nDuplicates: ", df.duplicated().sum())

df.drop_duplicates(inplace=True)


ynMapping = {'Yes': 1, 'No': 0}
sexMapping = {"Male": 1, "Female": 0}
rangeMapping = {"High": 2, "Normal": 1, "Low": 0}
correlationMapping = {"Positive": 1, "Negative": 0}


binary_columns = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
df[binary_columns] = df[binary_columns].replace(ynMapping)
df['Gender'] = df['Gender'].replace(sexMapping)
df[['Blood Pressure', 'Cholesterol Level']] = df[
	['Blood Pressure', 'Cholesterol Level']
].replace(rangeMapping)
df['Outcome Variable'] = df['Outcome Variable'].replace(correlationMapping)


df.to_csv(os.path.join(base_dir, "cleaned", "cleaned.csv"), index=False)
print(df.head())

