from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def importData():
    url = Path(__file__).resolve().parents[1] / "data" / "raw" / "mergedData.csv"
    balance_data = pd.read_csv(url, header=0)
    balance_data = balance_data.drop(columns=['Unnamed: 133'], errors='ignore')
    print("Dataset Length:", len(balance_data))
    print("Dataset Shape:", balance_data.shape)
    print("Dataset Head:", balance_data.head())
    return balance_data


def splitdataset(balance_data):
    feature_data = balance_data.drop("prognosis", axis=1)
    X = feature_data.to_numpy()
    le = LabelEncoder()
    Y = le.fit_transform(balance_data['prognosis'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=100
    )
    return X, Y, X_train, X_test, y_train, y_test


def main():
    balance_data = importData()
    print("Classes:", balance_data['prognosis'].unique())
    _, _, X_train, X_test, y_train, y_test = splitdataset(balance_data)

    model = BernoulliNB(alpha=0.5, fit_prior=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
