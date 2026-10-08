from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def importData():
    csv_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "raw"
        / "mergedData.csv"
    )
    data = pd.read_csv(csv_path)
    print("Dataset Length:", len(data))
    print("Dataset Shape:", data.shape)
    print("Dataset Head:\n", data.head())
    return data

def splitDataset(data):
    features = data.drop(columns=["prognosis"])


    disease_labels = data["prognosis"]
    print('no of labels:', data["prognosis"].nunique())
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        disease_labels,
        test_size=0.2,
        random_state=100,
        stratify=disease_labels,
    )
    return X_train, X_test, y_train, y_test


def train_using_kmeans(features, disease_labels, n_clusters=None):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    if n_clusters is None:
        n_clusters = disease_labels.nunique()

    kmeans = KMeans(n_clusters=n_clusters, random_state=100, n_init=10)
    cluster_labels = kmeans.fit_predict(features_scaled)
    cluster_disease = (
        pd.DataFrame(
            {"cluster": cluster_labels, "disease": disease_labels.to_numpy()}
        )
        .groupby("cluster")["disease"]
        .agg(lambda labels: labels.mode().iloc[0])
        .to_dict()
    )

    return kmeans, scaler, cluster_disease


def prediction(features, kmeans, scaler, cluster_disease):
    features_scaled = scaler.transform(features)
    cluster_labels = kmeans.predict(features_scaled)
    predicted_diseases = pd.Series(cluster_labels, index=features.index).map(
        cluster_disease
    )
    if predicted_diseases.isna().any():
        raise ValueError(
            "Some predicted clusters do not have a corresponding disease label."
        )
    print("Predicted Diseases:\n", predicted_diseases.to_numpy())
    return predicted_diseases


def cal_accuracy(y_test, y_pred):
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy * 100)
    print(
        "\nClassification Report:\n",
        classification_report(y_test, y_pred, zero_division=0),
    )
    return accuracy

def scatterPlot(features, disease_labels, kmeans, scaler, cluster_disease):
    features_scaled = scaler.transform(features)
    pca = PCA(n_components=3, random_state=100)
    components_3d = pca.fit_transform(features_scaled)

    cluster_ids = kmeans.predict(features_scaled)
    mapped_diseases = pd.Series(cluster_ids).map(cluster_disease)

    plot_df = pd.DataFrame(
        components_3d, 
        columns=["PCA1", "PCA2", "PCA3"], 
        index=features.index
    )

    plot_df["Cluster ID"] = cluster_ids.astype(str)
    plot_df["Assigned Cluster Disease"] = mapped_diseases
    plot_df["Actual Disease"] = disease_labels

    fig = px.scatter_3d(
        plot_df,
        x="PCA1",
        y="PCA2",
        z="PCA3",
        color="Assigned Cluster Disease",
        hover_data=["Cluster ID", "Actual Disease"],
        title="3D K-Means Symptom Clusters (Labeled by Assigned Disease)",
        opacity=0.7,
        width=900,
        height=700,
    )

    fig.show()

if __name__ == "__main__":
    data = importData()
    X_train, X_test, y_train, y_test = splitDataset(data)

    print("\n Training using K-Means")
    kmeans, scaler, cluster_disease = train_using_kmeans(X_train, y_train)
    y_pred = prediction(X_test, kmeans, scaler, cluster_disease)
    cal_accuracy(y_test, y_pred)
    scatterPlot(X_train, y_train, kmeans, scaler, cluster_disease)