import numpy as np
import pandas as pd

from sklearn.metrics import (confusion_matrix, accuracy_score, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

def importData():
    url = "../data/cleaned/Training.csv"
    balance_data = pd.read_csv(url, header=0)
    print("Dataset Length:", len(balance_data))
    print("Dataset Shape:", balance_data.shape)
    print("Dataset Head:", balance_data.head())
    return balance_data

def splitDataset(balance_data):
    balance_data = balance_data.drop(columns=['Unnamed: 133'], errors='ignore')
    feature_data = balance_data.drop("prognosis",axis=1)
    X = feature_data.to_numpy()
    le = LabelEncoder()
    Y = balance_data['prognosis']

    Y = le.fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=100)
    return X, Y, X_train, X_test, y_train, y_test

def train_using_knn(X_train, y_train, k=5):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    
    return knn, scaler

def prediction(X_test, knn, scaler):
    X_test_scaled = scaler.transform(X_test)
    y_pred = knn.predict(X_test_scaled)
    print("Predicted Values: \n", y_pred)
    
    return y_pred

def cal_accuracy(y_test, y_pred):
    print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("\nAccuracy: ", accuracy_score(y_test, y_pred) * 100 )
    print("\nClassification Report: \n", classification_report(y_test, y_pred))

def test_different_k_values(X_train, y_train, X_test, y_test, scaler, k_values):
        print(f"\nTesting different k values for KNN:")
        for k in k_values:
            knn, scaler = train_using_knn(X_train, y_train, k)
            y_pred = prediction(X_test, knn, scaler)
            cal_accuracy(y_test, y_pred)

if __name__ == "__main__":
    data = importData()
    X, Y, X_train, X_test, y_train, y_test = splitDataset(data)
    
    # Train KNN with default k=5
    knn, scaler = train_using_knn(X_train, y_train)
    y_pred = prediction(X_test, knn, scaler)
    cal_accuracy(y_test, y_pred)
    
    # # Test different k values
    # k_values = [1, 3, 5, 7, 9]
    # test_different_k_values(X_train, y_train, X_test, y_test, scaler, k_values)
