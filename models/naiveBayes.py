import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import GaussianNB, CategoricalNB

def importData():
    url = "../data/processed/mergedData.csv"
    balance_data = pd.read_csv(url, header=0)
    print("Dataset Length:", len(balance_data))
    print("Dataset Shape:", balance_data.shape)
    print("Dataset Head:", balance_data.head())
    return balance_data

def trainAndEvaluateMixedNB(df):
    y = df['Outcome Variable'].to_numpy()

    if not pd.api.types.is_numeric_dtype(df['Disease']):
        encoder = OrdinalEncoder()
        df['Disease'] = encoder.fit_transform(df[['Disease']]).astype(int)
    # Continuous feature: AGe
    X_continuous = df[['Age']].to_numpy()

    # Categorical feature: Disease, Fever, Cough, Fatigue, Difficulty Breathing, Gender, Blood Pressure, Cholesterol
    discrete_cols = ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level']
    X_discrete = df[discrete_cols].to_numpy()

    # Train-test split
    X_disc_train, X_disc_test, X_cont_train, X_cont_test, y_train, y_test = train_test_split(
        X_discrete, X_continuous, y, test_size=0.2, random_state=42, stratify=y
    )

    # Training Categorical Naive Bayes model
    model_categorical = CategoricalNB()
    model_categorical.fit(X_disc_train, y_train)

    # Training Gaussian Naive Bayes model
    model_gaussian = GaussianNB()
    model_gaussian.fit(X_cont_train, y_train)

    # Make predicitons

    log_prob_categorical = model_categorical.predict_log_proba(X_disc_test)
    log_prob_gaussian = model_gaussian.predict_log_proba(X_cont_test)

    # This part we combine the log probabilities from both models to make a final prediction
    combined_log_prob = log_prob_categorical + log_prob_gaussian

    # THen we select class with highest probability
    y_pred = model_categorical.classes_[np.argmax(combined_log_prob, axis=1)]

    # Output Results
    print("\n---Hybrid Naive Bayes Model Results ---")
    print("\n Accuracy Score: ", accuracy_score(y_test, y_pred))
    print("\n Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("\n Classification Report: \n", classification_report(y_test, y_pred))

if __name__ == "__main__":
        data = importData()
        trainAndEvaluateMixedNB(data)
 