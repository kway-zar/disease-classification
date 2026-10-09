from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def importData():
    csv_path = "mergedData.csv"
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

def column_names(data):
    data.columns = data.columns.str.strip()
    return list(data.columns)

def splitDataset(data):
    features = data.drop(columns=["prognosis"])
    disease_labels = data["prognosis"]
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

    return kmeans, scaler, cluster_disease, features_scaled, cluster_labels


def plot_kmeans_clusters_tsne(features_scaled, cluster_labels, kmeans, cluster_disease):
    """
    Uses t-SNE for non-linear dimensionality reduction and projects centroids cleanly into the 2D space.
    """
    n_samples = features_scaled.shape[0]

    all_points = np.vstack([features_scaled, kmeans.cluster_centers_])

    tsne = TSNE(n_components=2, perplexity=min(30, n_samples - 1), random_state=100)
    tsne_all = tsne.fit_transform(all_points)

    tsne_features = tsne_all[:n_samples]
    tsne_centroids = tsne_all[n_samples:]

    mapped_labels = np.array([f"Cluster {c}: {cluster_disease[c]}" for c in cluster_labels])

    plt.figure(figsize=(14, 8))
    sns.scatterplot(
        x=tsne_features[:, 0],
        y=tsne_features[:, 1],
        hue=mapped_labels,
        palette="tab20",
        alpha=0.6,
        s=80,
    )
    plt.scatter(
        tsne_centroids[:, 0],
        tsne_centroids[:, 1],
        c="red",
        marker="X",
        s=150,
        label="Cluster Centroids",
        edgecolor="black",
        zorder=5,
    )
    for idx, (x_pos, y_pos) in enumerate(tsne_centroids):
        plt.text(
            x_pos,
            y_pos,
            f"  {idx}",
            fontsize=9,
            weight="bold",
            color="black",
            zorder=6,
        )
    plt.title("K-Means Clustering Visualization (2D t-SNE Projection)", fontsize=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)

    # Place the legend horizontally below the plot
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        borderaxespad=0.0,
        fontsize=9
    )

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


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
    print("\nPredicted Diseases:\n", predicted_diseases.to_numpy())
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

if __name__ == "__main__":
    print("Column Names:", column_names(importData()))

    data = importData()
    X_train, X_test, y_train, y_test = splitDataset(data)

    print("\n Training using K-Means...")
    kmeans, scaler, cluster_disease, X_train_scaled, train_cluster_labels = train_using_kmeans(
        X_train, y_train
    )

    plot_kmeans_clusters_tsne(X_train_scaled, train_cluster_labels, kmeans, cluster_disease)

    y_pred = prediction(X_test, kmeans, scaler, cluster_disease)
    cal_accuracy(y_test, y_pred)

    print("\nTraining using K-Means")
    kmeans, scaler, cluster_disease, _, _ = train_using_kmeans(X_train, y_train)
    y_pred = prediction(X_test, kmeans, scaler, cluster_disease)
    cal_accuracy(y_test, y_pred)