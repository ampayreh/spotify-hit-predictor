# MSIS 522 HW1: Spotify Tracks — Classification Pipeline

## Dataset

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — 114,000 tracks with audio features (danceability, energy, loudness, acousticness, etc.) across 114 genres.

**Task:** Binary classification — predict `is_popular` (1 if popularity >= 50, else 0).

## Deployed App

[Streamlit Community Cloud URL — TBD after deployment]

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

To regenerate models from the notebook:

```bash
pip install nbclient nbformat ipykernel
python -c "
import nbformat
from nbclient import NotebookClient
with open('notebooks/analysis.ipynb') as f:
    nb = nbformat.read(f, as_version=4)
NotebookClient(nb, timeout=600, kernel_name='python3',
               resources={'metadata': {'path': '.'}}).execute()
with open('notebooks/analysis.ipynb', 'w') as f:
    nbformat.write(nb, f)
"
```

## Project Structure

```
msis522-hw1/
├── notebooks/
│   └── analysis.ipynb          # All EDA + modeling code (Parts 1-3 + Bonus)
├── models/                     # Saved models and artifacts
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   ├── mlp_model.keras
│   └── mlp_model.pkl
├── figures/                    # Generated visualizations
├── app.py                      # Streamlit app (Part 4)
├── mlp_wrapper.py              # Keras model wrapper for joblib
├── requirements.txt
├── README.md
└── data/
    └── dataset.csv
```

## Dataset Selection

Evaluated Spotify Tracks and NYC Airbnb datasets. Selected Spotify because it offers 114,000 rows with near-zero missing values, 12+ continuous audio features ideal for rich EDA visualizations and intuitive SHAP explanations.
