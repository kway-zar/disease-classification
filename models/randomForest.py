import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/cleaned/Healthcare_Processed.csv')

category_map = {
    'Common Cold': 'Respiratory', 'Influenza': 'Respiratory', 'COVID-19': 'Respiratory',
    'Pneumonia': 'Respiratory', 'Tuberculosis': 'Respiratory', 'Sinusitis': 'Respiratory', 'Bronchitis': 'Respiratory', 'Asthma': 'Respiratory',
    'Hypertension': 'Cardiovascular', 'Heart Disease': 'Cardiovascular', 'Stroke': 'Cardiovascular',
    'Migraine': 'Neurological', 'Epilepsy': 'Neurological', 'Dementia': 'Neurological', "Parkinson's": 'Neurological',
    'Depression': 'Mental Health', 'Anxiety': 'Mental Health',
    'Gastritis': 'Gastrointestinal', 'Food Poisoning': 'Gastrointestinal', 'IBS': 'Gastrointestinal', 'Liver Disease': 'Gastrointestinal', 'Ulcer': 'Gastrointestinal',
    'Diabetes': 'Metabolic', 'Thyroid Disorder': 'Metabolic', 'Obesity': 'Metabolic', 'Anemia': 'Metabolic',
    'Allergy': 'Immunological', 'Arthritis': 'Musculoskeletal', 'Dermatitis': 'Dermatological', 'Chronic Kidney Disease': 'Renal'
}

df['Disease_Category'] = df['Disease'].map(category_map)
 
X = df.drop(columns=['Patient_ID', 'Disease', 'Disease_Category'], errors='ignore')
y = df['Disease_Category']

X['symptom_count'] = X.sum(axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred, zero_division=0))
