import numpy as np
import pandas as pd

from .kMeans import DISEASE_CATEGORIES, KMeansModel, initialize_model


def normalize_symptom_name(name):
    return "".join(character for character in name.casefold() if character.isalnum())


def parse_symptoms(user_input, symptom_columns):
    symptom_lookup = {
        normalize_symptom_name(symptom): symptom for symptom in symptom_columns
    }
    selected = []
    unknown = []

    for entered in user_input:
        entered = entered.strip()
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


def predict_symptom_matches(symptoms, model: KMeansModel, top_n=3):
    symptom_vector = pd.DataFrame(
        [[int(symptom in symptoms) for symptom in model.symptom_columns]],
        columns=model.symptom_columns,
    )
    selected_features = [
        index
        for index, symptom in enumerate(model.symptom_columns)
        if symptom in symptoms
    ]
    scaled_symptoms = model.scaler.transform(symptom_vector)[0, selected_features]
    distances = np.linalg.norm(
        model.kmeans.cluster_centers_[:, selected_features] - scaled_symptoms,
        axis=1,
    )

    matches = []
    seen_diseases = set()
    for cluster in np.argsort(distances):
        disease = model.cluster_disease.get(int(cluster))
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

    return causes, courses, systems


def analyze(user_input, model=None):
    """Analyze symptom names or numbers with an initialized model."""
    if model is None:
        model = initialize_model()
    symptoms = parse_symptoms(user_input, model.symptom_columns)
    matches = predict_symptom_matches(symptoms, model)
    summary = summarize_match_categories(matches)
    return summary, matches, model.validation_accuracy