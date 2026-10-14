from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "data" / "processed" / "mergedData.csv"


@dataclass
class KMeansModel:
    kmeans: KMeans
    scaler: StandardScaler
    cluster_disease: dict
    symptom_columns: list[str]
    validation_accuracy: float

DISEASE_CATEGORIES = {
    "(vertigo) Paroymsal  Positional Vertigo": {
        "etiology": "Varies; vertigo is a symptom, not a single disease cause",
        "course": "Often episodic; depends on the underlying cause",
        "system": "Neurological / vestibular",
    },
    "AIDS": {
        "etiology": "Infectious (HIV)",
        "course": "Chronic",
        "system": "Immune",
    },
    "Acne": {
        "etiology": "Non-communicable (multiple contributing factors)",
        "course": "Often chronic or recurrent",
        "system": "Integumentary (skin)",
    },
    "Alcoholic hepatitis": {
        "etiology": "Non-communicable (alcohol-related liver injury)",
        "course": "Can be acute or chronic",
        "system": "Gastrointestinal / hepatobiliary",
    },
    "Allergy": {
        "etiology": "Allergic / immune-mediated",
        "course": "Variable",
        "system": "Varies by affected organ",
    },
    "Arthritis": {
        "etiology": "Varies by arthritis type",
        "course": "Often chronic",
        "system": "Musculoskeletal",
    },
    "Bronchial Asthma": {
        "etiology": "Non-communicable (multifactorial)",
        "course": "Chronic, with episodes",
        "system": "Respiratory",
    },
    "Cervical spondylosis": {
        "etiology": "Degenerative",
        "course": "Usually chronic",
        "system": "Musculoskeletal / neurological",
    },
    "Chicken pox": {
        "etiology": "Infectious (varicella-zoster virus)",
        "course": "Usually acute",
        "system": "Integumentary / systemic",
    },
    "Chronic cholestasis": {
        "etiology": "Varies by underlying cause",
        "course": "Chronic",
        "system": "Gastrointestinal / hepatobiliary",
    },
    "Common Cold": {
        "etiology": "Infectious (usually viral)",
        "course": "Usually acute",
        "system": "Respiratory",
    },
    "Dengue": {
        "etiology": "Infectious (dengue virus)",
        "course": "Usually acute",
        "system": "Systemic / circulatory",
    },
    "Diabetes": {
        "etiology": "Metabolic",
        "course": "Chronic",
        "system": "Endocrine / metabolic",
    },
    "Dimorphic hemmorhoids(piles)": {
        "etiology": "Non-communicable",
        "course": "Variable; may recur",
        "system": "Gastrointestinal / vascular",
    },
    "Drug Reaction": {
        "etiology": "Medication-related adverse reaction",
        "course": "Variable",
        "system": "Varies by reaction",
    },
    "Fungal infection": {
        "etiology": "Infectious (fungal)",
        "course": "Variable",
        "system": "Often integumentary (skin)",
    },
    "GERD": {
        "etiology": "Non-communicable",
        "course": "Often chronic or recurrent",
        "system": "Gastrointestinal",
    },
    "Gastroenteritis": {
        "etiology": "Often infectious; other causes are possible",
        "course": "Usually acute",
        "system": "Gastrointestinal",
    },
    "Heart attack": {
        "etiology": "Non-communicable (usually coronary artery disease)",
        "course": "Acute medical emergency",
        "system": "Cardiovascular",
    },
    "Hepatitis B": {
        "etiology": "Infectious (hepatitis B virus)",
        "course": "May be acute or chronic",
        "system": "Gastrointestinal / hepatobiliary",
    },
    "Hepatitis C": {
        "etiology": "Infectious (hepatitis C virus)",
        "course": "May be acute or chronic",
        "system": "Gastrointestinal / hepatobiliary",
    },
    "Hepatitis D": {
        "etiology": "Infectious (hepatitis D virus; occurs with hepatitis B)",
        "course": "May be acute or chronic",
        "system": "Gastrointestinal / hepatobiliary",
    },
    "Hepatitis E": {
        "etiology": "Infectious (hepatitis E virus)",
        "course": "Usually acute",
        "system": "Gastrointestinal / hepatobiliary",
    },
    "Hypertension": {
        "etiology": "Usually non-communicable (multiple contributing factors)",
        "course": "Usually chronic",
        "system": "Cardiovascular",
    },
    "Hyperthyroidism": {
        "etiology": "Metabolic / endocrine; cause varies",
        "course": "Variable; often persistent without treatment",
        "system": "Endocrine / metabolic",
    },
    "Hypoglycemia": {
        "etiology": "Metabolic; cause varies",
        "course": "Often acute episodes",
        "system": "Endocrine / metabolic",
    },
    "Hypothyroidism": {
        "etiology": "Metabolic / endocrine; cause varies",
        "course": "Usually chronic",
        "system": "Endocrine / metabolic",
    },
    "Impetigo": {
        "etiology": "Infectious (bacterial)",
        "course": "Usually acute",
        "system": "Integumentary (skin)",
    },
    "Jaundice": {
        "etiology": "Not classifiable from this label; jaundice is a sign with many causes",
        "course": "Depends on the underlying cause",
        "system": "Gastrointestinal / hepatobiliary",
    },
    "Malaria": {
        "etiology": "Infectious (Plasmodium parasite)",
        "course": "Usually acute episodes; may recur",
        "system": "Systemic / circulatory",
    },
    "Migraine": {
        "etiology": "Non-communicable neurological disorder",
        "course": "Chronic with episodic attacks",
        "system": "Neurological",
    },
    "Osteoarthristis": {
        "etiology": "Degenerative",
        "course": "Chronic",
        "system": "Musculoskeletal",
    },
    "Paralysis (brain hemorrhage)": {
        "etiology": "Non-communicable (brain hemorrhage / stroke)",
        "course": "Acute onset; effects may persist",
        "system": "Neurological",
    },
    "Peptic ulcer diseae": {
        "etiology": "Varies; may involve H. pylori infection or medication",
        "course": "Variable; may recur",
        "system": "Gastrointestinal",
    },
    "Pneumonia": {
        "etiology": "Often infectious; cause may be bacterial, viral, or other",
        "course": "Often acute; duration varies",
        "system": "Respiratory",
    },
    "Psoriasis": {
        "etiology": "Immune-mediated (not contagious)",
        "course": "Chronic, often with flares",
        "system": "Integumentary (skin)",
    },
    "Tuberculosis": {
        "etiology": "Infectious (Mycobacterium tuberculosis)",
        "course": "Usually prolonged; can be active or latent",
        "system": "Often respiratory; can affect other systems",
    },
    "Typhoid": {
        "etiology": "Infectious (Salmonella Typhi)",
        "course": "Usually acute",
        "system": "Gastrointestinal / systemic",
    },
    "Urinary tract infection": {
        "etiology": "Usually infectious (often bacterial)",
        "course": "Often acute; may recur",
        "system": "Urinary",
    },
    "Varicose veins": {
        "etiology": "Non-communicable (venous valve dysfunction)",
        "course": "Usually chronic",
        "system": "Cardiovascular / vascular",
    },
    "hepatitis A": {
        "etiology": "Infectious (hepatitis A virus)",
        "course": "Usually acute",
        "system": "Gastrointestinal / hepatobiliary",
    },
}


def importData():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    data = data.drop(columns=["Unnamed: 133"], errors="ignore")
    data["prognosis"] = data["prognosis"].str.strip()
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

    kmeans = KMeans(n_clusters=n_clusters, random_state=100, n_init=1)
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


def evaluate_kmeans_accuracy(data):
    X_train, X_test, y_train, y_test = splitDataset(data)
    kmeans, scaler, cluster_disease, _, _ = train_using_kmeans(X_train, y_train)
    clusters = kmeans.predict(scaler.transform(X_test))
    predictions = pd.Series(clusters, index=y_test.index).map(cluster_disease)
    if predictions.isna().any():
        raise ValueError("A test prediction mapped to a cluster without a disease label.")
    return accuracy_score(y_test, predictions)


def initialize_model(data=None):
    """Load the dataset, train K-Means, and return reusable model state."""
    if data is None:
        data = importData()

    symptom_columns = data.drop(columns=["prognosis"]).columns.tolist()
    validation_accuracy = evaluate_kmeans_accuracy(data)
    kmeans, scaler, cluster_disease, _, _ = train_using_kmeans(
        data[symptom_columns], data["prognosis"]
    )
    return KMeansModel(
        kmeans=kmeans,
        scaler=scaler,
        cluster_disease=cluster_disease,
        symptom_columns=symptom_columns,
        validation_accuracy=validation_accuracy,
    )


# def plot_kmeans_clusters_tsne(features_scaled, cluster_labels, kmeans, cluster_disease):
#     """
#     Uses t-SNE for non-linear dimensionality reduction and projects centroids cleanly into the 2D space.
#     """
#     n_samples = features_scaled.shape[0]

#     all_points = np.vstack([features_scaled, kmeans.cluster_centers_])

#     tsne = TSNE(n_components=2, perplexity=min(30, n_samples - 1), random_state=100)
#     tsne_all = tsne.fit_transform(all_points)

#     tsne_features = tsne_all[:n_samples]
#     tsne_centroids = tsne_all[n_samples:]

#     mapped_labels = np.array([f"Cluster {c}: {cluster_disease[c]}" for c in cluster_labels])

#     plt.figure(figsize=(14, 8))
#     sns.scatterplot(
#         x=tsne_features[:, 0],
#         y=tsne_features[:, 1],
#         hue=mapped_labels,
#         palette="tab20",
#         alpha=0.6,
#         s=80,
#     )
#     plt.scatter(
#         tsne_centroids[:, 0],
#         tsne_centroids[:, 1],
#         c="red",
#         marker="X",
#         s=150,
#         label="Cluster Centroids",
#         edgecolor="black",
#         zorder=5,
#     )
#     for idx, (x_pos, y_pos) in enumerate(tsne_centroids):
#         plt.text(
#             x_pos,
#             y_pos,
#             f"  {idx}",
#             fontsize=9,
#             weight="bold",
#             color="black",
#             zorder=6,
#         )
#     plt.title("K-Means Clustering Visualization (2D t-SNE Projection)", fontsize=14)
#     plt.xlabel("t-SNE Dimension 1", fontsize=12)
#     plt.ylabel("t-SNE Dimension 2", fontsize=12)

#     # Place the legend horizontally below the plot
#     plt.legend(
#         loc="upper center",
#         bbox_to_anchor=(0.5, -0.15),
#         ncol=4,
#         borderaxespad=0.0,
#         fontsize=9
#     )

#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.tight_layout()
#     if plt.get_backend().lower() != "agg":
#         plt.show()
#     plt.close()


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

def normalize_symptom_name(name):
    return "".join(character for character in name.casefold() if character.isalnum())


def parse_symptoms(user_input, symptom_columns):
    symptom_lookup = {
        normalize_symptom_name(symptom): symptom for symptom in symptom_columns
    }
    entered_symptoms = user_input
    selected = []
    unknown = []

    for entered in entered_symptoms:
        if not entered:
            continue
        if entered.isdecimal():
            symptom_index = int(entered) - 1
            if 0 <= symptom_index < len(symptom_columns):
                matched = symptom_columns[symptom_index]
            else:
                unknown.append(entered)
                continue
        else:
            matched = symptom_lookup.get(normalize_symptom_name(entered))
        if matched is None:
            unknown.append(entered)
        elif matched not in selected:
            selected.append(matched)

    if unknown:
        raise ValueError(
            "Unrecognized symptom(s): "
            + ", ".join(unknown)
            + ". Type 'list' to see symptom names and their number choices."
        )
    if not selected:
        raise ValueError("Enter at least one symptom.")

    return selected


def predict_symptom_matches(symptoms, symptom_columns, kmeans, scaler, cluster_disease, top_n=3):
    symptom_vector = pd.DataFrame(
        [[int(symptom in symptoms) for symptom in symptom_columns]],
        columns=symptom_columns,
    )
    selected_features = [
        index for index, symptom in enumerate(symptom_columns) if symptom in symptoms
    ]
    scaled_symptoms = scaler.transform(symptom_vector)[0, selected_features]
    distances = np.linalg.norm(
        kmeans.cluster_centers_[:, selected_features] - scaled_symptoms,
        axis=1,
    )

    matches = []
    seen_diseases = set()
    for cluster in np.argsort(distances):
        disease = cluster_disease.get(int(cluster))
        if disease is not None and disease not in seen_diseases:
            matches.append((disease, float(distances[cluster])))
            seen_diseases.add(disease)
        if len(matches) == top_n:
            break

    return matches


def summarize_match_categories(matches):
    categories = [DISEASE_CATEGORIES.get(disease, {}) for disease, _ in matches]
    causes = set()
    courses = set()
    systems = set()

    for category in categories:
        etiology = category.get("etiology", "").casefold()
        if "infectious" in etiology:
            causes.add("Infectious")
        elif "degenerative" in etiology:
            causes.add("Degenerative")
        elif "metabolic" in etiology:
            causes.add("Metabolic")
        elif "immune-mediated" in etiology or "allergic" in etiology:
            causes.add("Immune/allergic")
        elif "non-communicable" in etiology:
            causes.add("Non-communicable")
        else:
            causes.add("Varies or is not specified")

        course = category.get("course", "").casefold()
        if "chronic" in course and "acute" in course:
            courses.add("Acute or chronic/variable")
        elif "chronic" in course:
            courses.add("Chronic")
        elif "acute" in course:
            courses.add("Acute")
        else:
            courses.add("Variable or depends on cause")
        if category.get("system"):
            systems.add(category["system"])

    print("\nCategories represented among the closest matches:")
    print("  By cause: " + ", ".join(sorted(causes)))
    print("  By duration/course: " + ", ".join(sorted(courses)))
    print("  By body system: " + ", ".join(sorted(systems)))
    print(
        "  By spread: Not determined; this depends on location and current "
        "outbreak data."
    )
    return causes, courses, systems


def show_symptom_menu(symptom_columns):
    print("\nSupported symptoms (enter a number or name):")
    for index, symptom in enumerate(symptom_columns, start=1):
        print(f"  {index:>3}. {symptom.replace('_', ' ')}")



def start(user_input):
    from .input import analyze

    return analyze(user_input)

    # print("\nClosest disease matches (distance is not a confidence score):")
    # for rank, (disease, distance) in enumerate(matches, start=1):
    #     print(f"  {rank}. {disease} (relative distance: {distance:.2f})")

    # print(
    #         f"\nModel held-out accuracy: {validation_accuracy:.1%}. This measures "
    #         "performance on dataset records, not confidence in your result. "
    #         "Distances are not probabilities. This educational tool cannot "
    #         "diagnose disease; consult a qualified healthcare professional "
    #         "about symptoms."
    #     )


