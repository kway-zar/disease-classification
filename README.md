# Disease Classification

Disease Classification is a machine-learning project that explores whether a
set of reported symptoms can be matched to likely diseases. The main user
interface is a Streamlit application: users select symptoms, the application
compares that binary symptom vector with clusters learned from the dataset, and
then displays the three closest disease matches.

This is an educational data-science project, not a medical diagnostic tool.
Its results are predictions from a small, fixed dataset and should not be used
to make health decisions. Anyone with concerning symptoms should consult a
qualified healthcare professional.

## What the application does

When the Streamlit app starts, it:

1. Loads `data/cleaned/mergedData.csv`.
2. Removes the optional `Unnamed: 133` CSV index column and trims the disease
     labels in the `prognosis` column.
3. Treats every remaining column as a binary symptom feature.
4. Standardizes the feature values and trains a scikit-learn K-Means model.
5. Assigns each cluster its most common disease label.
6. Evaluates the model on a stratified 80/20 validation split.

In the interface, select one or more symptoms and click **Analyze**. The
results include:

- The three closest disease matches and their relative distances.
- A summary of the matches by cause, expected course, and body system.
- The validation accuracy calculated when the model is initialized.

The model is cached with Streamlit's `st.cache_resource`, so it is trained once
per application process instead of on every interaction.

## Project structure

```text
.
├── main.py                         # Streamlit application entry point
├── data/
│   ├── raw/                        # Source CSV files
│   ├── cleaned/mergedData.csv      # Dataset used by the app and models
│   ├── cleaner.py                  # Cleaning for patient-profile data
│   └── cleanerHealthCare.py        # Symptom-column preprocessing
├── models/
│   ├── kMeans.py                   # Main model, evaluation, and metadata
│   ├── input.py                    # Symptom parsing and nearest-match logic
│   ├── bernoulliNaiveBayes.py      # Bernoulli Naive Bayes experiment
│   ├── naiveBayes.py               # Mixed categorical/Gaussian NB experiment
│   └── decisionTree.py             # Gini and entropy decision-tree experiment
├── disease_classification/         # Python package and Streamlit component
│   └── frontend/                   # React/TypeScript component source
└── test/                           # Pytest unit tests
```

## Requirements

- Python 3.10 or newer
- Node.js 24 or newer only when rebuilding the frontend component
- `uv` is recommended for Python environment and package management

The Python dependencies are declared in `pyproject.toml`, including Streamlit
and scikit-learn. The model code also uses NumPy and pandas.

## Installation

From the repository root, create or activate a virtual environment and install
the project in editable mode:

```sh
uv venv
uv pip install -e .
```

If your environment does not already provide the data-science dependencies,
install them explicitly:

```sh
uv pip install numpy pandas
```

## Run the application

Start the main Streamlit interface from the repository root:

```sh
uv run streamlit run main.py
```

The application expects `data/cleaned/mergedData.csv` to exist at that exact
path. The path is resolved relative to the repository by `models/kMeans.py`.

## Run the standalone model experiments

The Bernoulli Naive Bayes experiment can be run from the repository root:

```sh
uv run python models/bernoulliNaiveBayes.py
```

The Decision Tree script uses a relative dataset path, so run it from the
`models/` directory:

```sh
cd models
uv run --with matplotlib python decisionTree.py
cd ..
```

`models/kMeans.py` exposes reusable functions such as `initialize_model`,
`evaluate_kmeans_accuracy`, and `prediction`; the Streamlit app uses the first
of these through `main.py`. `models/naiveBayes.py` contains a separate mixed
categorical/Gaussian experiment and expects columns from a different patient
profile dataset, so it is not part of the main app startup path.

## Classification report results

The following results were generated from the current
`data/cleaned/mergedData.csv` dataset. It contains 4,962 rows, 132 symptom
features, 41 disease classes, and a 993-row test set. The reported values are
the aggregate rows from scikit-learn's `classification_report`; they summarize
the per-class results rather than replacing them.

| Model | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Decision Tree (Gini) | 100.00% | 1.00 | 1.00 | 1.00 | 1.00 |
| Decision Tree (Entropy) | 100.00% | 1.00 | 1.00 | 1.00 | 1.00 |
| Bernoulli Naive Bayes | 100.00% | 1.00 | 1.00 | 1.00 | 1.00 |
| K-Means | 97.48% | 0.9632 | 0.9746 | 0.9667 | 0.9670 |

The Decision Tree and Bernoulli Naive Bayes reports contain 41 encoded disease
classes, each with precision, recall, and F1 equal to 1.00 on this split. The
K-Means report has lower aggregate precision because cluster assignments are
mapped to disease labels after unsupervised training. Its full per-class
report is printed by the K-Means evaluation code through
`classification_report`.

The mixed model in `models/naiveBayes.py` has no result in this table. It
expects an `Outcome Variable` target and patient-profile columns such as
`Age`, `Disease`, and `Gender`, while the current dataset uses `prognosis` and
binary symptom columns. Running it against the current file raises a missing
column error before training.

## Data preparation

The preprocessing scripts under `data/` convert source healthcare records into
model-ready CSV files. In particular, `cleanerHealthCare.py` expands the
comma-separated `Symptoms` field into one binary column per symptom. The main
application currently reads the already-prepared
`data/cleaned/mergedData.csv`; run or adapt the cleaning scripts before
replacing that file with a new dataset.

## Frontend component and packaging

The `disease_classification/frontend` directory is a React/TypeScript
Streamlit component scaffold. Its current implementation is the generated
example component and is separate from the symptom-analysis UI in `main.py`.
To rebuild its packaged assets:

```sh
cd disease_classification/frontend
npm install
npm run build
cd ../..
```

To build the Python wheel from the project root:

```sh
uv build
```

The wheel includes the compiled frontend assets under
`disease_classification/frontend/build`.

## Current limitations

- K-Means is an unsupervised algorithm; cluster labels are assigned after
    training using the most common disease in each cluster.
- The displayed validation accuracy is based on one fixed random split and is
    not a guarantee of real-world performance.
- The application does not collect patient history, severity, timing, age, or
    other clinical context.
- Unit tests for parsing and model behavior are in `test/test_models.py` and
    can be run with `uv run pytest test/test_models.py -q`.

