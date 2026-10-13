import numpy as np
import pandas as pd
import pytest

from models.input import (
    normalize_symptom_name,
    parse_symptoms,
    predict_symptom_matches,
    summarize_match_categories,
)
from models.kMeans import train_using_kmeans


def test_normalize_symptom_name_ignores_case_spaces_and_punctuation():
    assert normalize_symptom_name("Chest Pain") == "chestpain"
    assert normalize_symptom_name("  shortness-of-breath ") == "shortnessofbreath"


def test_parse_symptoms_accepts_names_and_one_based_numbers_without_duplicates():
    symptom_columns = ["Chest Pain", "Shortness of Breath", "Fatigue"]

    selected = parse_symptoms(
        [" chest-pain ", "2", "Fatigue", "1"], symptom_columns
    )

    assert selected == ["Chest Pain", "Shortness of Breath", "Fatigue"]


def test_parse_symptoms_rejects_unknown_symptoms():
    with pytest.raises(ValueError, match=r"Unrecognized symptom\(s\): nausea"):
        parse_symptoms(["nausea"], ["Chest Pain"])


def test_parse_symptoms_rejects_empty_input():
    with pytest.raises(ValueError, match="Enter at least one symptom"):
        parse_symptoms(["", "  "], ["Chest Pain"])


def test_summarize_match_categories_groups_known_disease_metadata():
    summary = summarize_match_categories(
        [("Common Cold", 0.1), ("Diabetes", 0.2), ("Cervical spondylosis", 0.3)]
    )

    causes, courses, systems = summary
    assert causes == {"Infectious", "Metabolic", "Degenerative"}
    assert courses == {"Acute", "Chronic"}
    assert systems == {
        "Respiratory",
        "Endocrine / metabolic",
        "Musculoskeletal / neurological",
    }


def test_train_using_kmeans_returns_cluster_labels_for_each_training_row():
    features = pd.DataFrame(
        {
            "fever": [1, 1, 0, 0, 0, 0],
            "cough": [1, 0, 1, 1, 0, 0],
        }
    )
    diseases = pd.Series(
        ["Common Cold", "Common Cold", "Pneumonia", "Pneumonia", "Diabetes", "Diabetes"]
    )

    kmeans, scaler, cluster_disease, features_scaled, cluster_labels = (
        train_using_kmeans(features, diseases, n_clusters=3)
    )

    assert features_scaled.shape == features.shape
    assert cluster_labels.shape == (len(features),)
    assert set(cluster_disease.values()).issubset(set(diseases))
    assert kmeans.n_clusters == 3
    np.testing.assert_allclose(scaler.transform(features), features_scaled)


def test_predict_symptom_matches_returns_unique_ranked_diseases():
    features = pd.DataFrame(
        {
            "fever": [1, 1, 0, 0, 0, 0],
            "cough": [1, 0, 1, 1, 0, 0],
        }
    )
    diseases = pd.Series(
        ["Common Cold", "Common Cold", "Pneumonia", "Pneumonia", "Diabetes", "Diabetes"]
    )
    kmeans, scaler, cluster_disease, _, _ = train_using_kmeans(
        features, diseases, n_clusters=3
    )

    model = type(
        "TestModel",
        (),
        {
            "kmeans": kmeans,
            "scaler": scaler,
            "cluster_disease": cluster_disease,
            "symptom_columns": list(features.columns),
        },
    )()

    matches = predict_symptom_matches(["fever"], model, top_n=2)

    assert len(matches) == 2
    assert len({disease for disease, _ in matches}) == len(matches)
    assert all(distance >= 0 for _, distance in matches)