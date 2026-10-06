import numpy as np
import pandas as pd
from pandas.core.common import random_state

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt


def importData():
    url = "../data/cleaned/Training.csv"
    balance_data = pd.read_csv(url, header=0)
    balance_data = balance_data.drop(columns=['Unnamed: 133'], errors='ignore')
    print("Dataset Length:", len(balance_data))
    print("Dataset Shape:", balance_data.shape)
    print("Dataset Head:", balance_data.head())
    return balance_data


def splitdataset(balance_data):
    feature_data = balance_data.drop("prognosis",axis=1)
    X = feature_data.to_numpy()
    le = LabelEncoder()
    Y = balance_data['prognosis']
    Y = le.fit_transform(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
    return X, Y, X_train, X_test, y_train, y_test

def train_using_gini(X_train, y_train):
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state = 42)
    clf_gini.fit(X_train, y_train)
    
    return clf_gini

def train_using_entropy(X_train, y_train):
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state = 42)
    clf_entropy.fit(X_train, y_train)

    return clf_entropy

def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted Values: \n", y_pred)
    
    return y_pred


def cal_accuracy(y_test, y_pred):
    print("\nCofusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("\nAccuracy: ", accuracy_score(y_test, y_pred) * 100)
    print("\nClassification Report: \n", classification_report(y_test, y_pred))

def plot_decision_tree(clf_object, feature_names, class_names):
    plt.figure(figsize=(15,10))
    plot_tree(clf_object, filled= True, feature_names = feature_names, class_names = class_names, rounded = True)

    plt.show()


if __name__ == "__main__":
 
    data = importData()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)

  
    print("\n----- Training Using Gini -----")
    clf_gini = train_using_gini(X_train, y_train)
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)

    
    print("\n----- Training Using Entropy -----")
    clf_entropy = train_using_entropy(X_train, y_train)
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
