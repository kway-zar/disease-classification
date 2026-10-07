# disease-classification

Streamlit component that allows you to do X

## Installation instructions

```sh
uv pip install disease-classification
```

### Development install (editable)

When developing this component locally, install it in editable mode so Streamlit picks up code changes without rebuilding a wheel. Run this from the directory that contains `pyproject.toml`:

```sh
uv pip install -e . --force-reinstall
```

## Usage instructions

```python
import streamlit as st

from disease_classification import disease_classification

value = disease_classification()

st.write(value)
```

## Build a wheel

To package this component for distribution:

1. Build the frontend assets (from `disease_classification/frontend`):

   ```sh
   npm i
   npm run build
   ```

2. Build the Python wheel using UV (from the project root):
   ```sh
   uv build
   ```

This will create a `dist/` directory containing your wheel. The wheel includes the compiled frontend from `disease_classification/frontend/build`.

### Requirements

- Python >= 3.10
- Node.js >= 24 (LTS)

### Expected output

- `dist/disease_classification-0.0.1-py3-none-any.whl`
- If you run `uv run --with build python -m build` (without `--wheel`), you’ll also get an sdist: `dist/disease-classification-0.0.1.tar.gz`

### Results
- Decision Tree:
    ```sh
     Gini: 100%
     Entropy: 100%

    ```
- Bernoulli Naive Bayes:
    ```sh
    Accuracy:100%
    ```
- KMeans:
    ```sh
    n=4 Accuracy:97.48%
    n=10 Accuracy: 97.48%

    ```
- KNN:
    ```sh
    k=5 Accuracy: 100%
    k=10 Accuracy: 99.70%
    ```

