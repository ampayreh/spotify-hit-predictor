"""MSIS 522 HW1 — Spotify Track Popularity Classifier (Streamlit App)."""

import json
import os

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st

matplotlib.use("Agg")
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MSIS 522 — Spotify Popularity Classifier",
    page_icon="🎵",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODELS_DIR = "models"
DATA_PATH = "data/dataset.csv"
FIGURES_DIR = "figures"

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]
NUM_FEATURES = AUDIO_FEATURES + ["duration_ms"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost_model.pkl",
}


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["is_popular"] = (df["popularity"] >= 50).astype(int)
    return df


@st.cache_resource
def load_models():
    models = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models


@st.cache_resource
def load_mlp():
    import tensorflow as tf
    path = os.path.join(MODELS_DIR, "mlp_model.keras")
    pre_path = os.path.join(MODELS_DIR, "mlp_preprocessor.pkl")
    if os.path.exists(path) and os.path.exists(pre_path):
        return tf.keras.models.load_model(path), joblib.load(pre_path)
    return None, None


@st.cache_data
def load_metrics():
    path = os.path.join(MODELS_DIR, "metrics_summary.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_data
def load_best_params():
    path = os.path.join(MODELS_DIR, "best_params.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_train_test_info():
    path = os.path.join(MODELS_DIR, "train_test_info.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


@st.cache_data
def load_shap_top():
    path = os.path.join(MODELS_DIR, "shap_top_features.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_shap_background():
    path = os.path.join(MODELS_DIR, "shap_background.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ---------------------------------------------------------------------------
# Load everything at startup
# ---------------------------------------------------------------------------
df = load_data()
sklearn_models = load_models()
mlp_model, mlp_preprocessor = load_mlp()
metrics_df = load_metrics()
best_params = load_best_params()
train_test_info = load_train_test_info()
shap_top = load_shap_top()
shap_bg = load_shap_background()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Summary",
    "Descriptive Analytics",
    "Model Performance",
    "Explainability & Prediction",
])

# ===================================================================
# TAB 1 — Executive Summary
# ===================================================================
with tab1:
    st.header("Executive Summary")

    st.markdown("""
### The Dataset

The Spotify Tracks dataset contains **114,000 tracks** sourced from the Spotify
Web API, spanning 114 distinct musical genres. Each track is described by a rich
set of audio features — including danceability, energy, loudness, speechiness,
acousticness, instrumentalness, liveness, valence, and tempo — that Spotify
computes from audio signal analysis. Alongside these continuous audio descriptors,
the dataset includes metadata such as track duration, whether a track is marked
explicit, the musical key and mode (major/minor), time signature, and genre
classification. Our target variable, `is_popular`, is a binary label that equals 1
when a track's popularity score (a 0–100 metric reflecting recent streaming
activity) reaches or exceeds 50, placing it roughly in the top quartile of all
tracks.

### Why This Matters

Understanding what makes a song popular has real commercial value. Record labels
invest millions in A&R (Artist & Repertoire) — the process of discovering and
developing talent — and rely heavily on intuition. A data-driven model that can
identify which audio characteristics distinguish hit-potential tracks from the
rest provides an objective lens for scouting decisions. Playlist curators at
streaming platforms can use the same insights to surface tracks that match the
profile of songs listeners already love. Music producers and artists can also
benefit: knowing that certain production choices (higher energy, louder mastering,
dance-oriented rhythms) correlate with greater popularity can inform creative
decisions without replacing artistic instinct.

### Our Approach

We followed a three-phase analytical pipeline. First, **descriptive analytics**
explored the distributions and relationships among audio features, confirming that
popular tracks cluster in a distinct region of the feature space characterized by
high energy, high loudness, high danceability, and low acousticness. Second,
**predictive modeling** benchmarked five classifiers — Logistic Regression,
Decision Tree, Random Forest, XGBoost, and a Multi-Layer Perceptron neural
network — using stratified cross-validation and grid search. All models used
class-weight balancing to account for the 74/26 imbalance in our target variable.
Third, **SHAP explainability analysis** on the best-performing tree-based model
provided both global and local feature-importance insights.

### Key Findings

XGBoost emerged as the top performer, achieving the highest F1 score (0.564) and
AUC-ROC (0.792) on the held-out test set. SHAP analysis revealed that **genre** is
the single most important predictor — whether a track belongs to a mainstream genre
(pop, hip-hop, k-pop) versus a niche genre is overwhelmingly predictive. Among
audio features, **energy, valence, and acousticness** are the three most
influential: high-energy, upbeat (high-valence), non-acoustic tracks are far more
likely to be popular. These findings translate directly into actionable insights for
industry stakeholders, offering a transparent, explainable framework for
understanding what makes music resonate with large audiences.
    """)

# ===================================================================
# TAB 2 — Descriptive Analytics
# ===================================================================
with tab2:
    st.header("Descriptive Analytics")

    # --- Target Distribution ---
    st.subheader("Target Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["is_popular"].value_counts().sort_index()
    sns.countplot(x="is_popular", data=df, ax=ax, palette=["#4C72B0", "#DD8452"])
    for i, v in enumerate(counts):
        ax.text(i, v + 500, f"{v:,}\n({v / len(df) * 100:.1f}%)", ha="center", fontsize=11)
    ax.set_xticklabels(["Not Popular (0)", "Popular (1)"])
    ax.set_title("Target Distribution: is_popular (popularity >= 50)")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "The target variable is moderately imbalanced with approximately 74% of tracks "
        "classified as not popular and 26% as popular. This imbalance is accounted for "
        "in our models using class-weight balancing and stratified splitting."
    )

    st.markdown("---")

    # --- Boxplots ---
    st.subheader("Audio Features by Popularity Class")
    top_feats = ["danceability", "energy", "loudness", "acousticness", "valence", "speechiness"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feat in zip(axes.ravel(), top_feats):
        sns.boxplot(x="is_popular", y=feat, data=df, ax=ax, palette=["#4C72B0", "#DD8452"])
        ax.set_xticklabels(["Not Popular", "Popular"])
        ax.set_title(feat.title())
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Popular tracks exhibit higher energy, danceability, and loudness with tighter "
        "distributions. Acousticness shows a clear downward shift for popular tracks, "
        "indicating that heavily produced (non-acoustic) music tends to gain more streams."
    )

    st.markdown("---")

    # --- Violin plots ---
    st.subheader("Distribution Shape of Key Features")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, feat in zip(axes, ["energy", "danceability", "acousticness"]):
        sns.violinplot(x="is_popular", y=feat, data=df, ax=ax,
                       palette=["#4C72B0", "#DD8452"], inner="quartile", cut=0)
        ax.set_xticklabels(["Not Popular", "Popular"])
        ax.set_title(feat.title())
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Violin plots reveal the full distribution shape. Energy for popular tracks "
        "concentrates in the 0.5-0.9 range, danceability clusters around 0.6-0.8, and "
        "acousticness is heavily right-skewed near 0 for popular tracks."
    )

    st.markdown("---")

    # --- Genre popularity rate ---
    st.subheader("Top 15 Genres by Popularity Rate")
    genre_rate = (
        df.groupby("track_genre")["is_popular"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=genre_rate["mean"], y=genre_rate.index, ax=ax, palette="viridis")
    ax.set_xlabel("Proportion of Popular Tracks")
    ax.set_ylabel("")
    ax.set_title("Top 15 Genres by Popularity Rate")
    for i, (val, _) in enumerate(genre_rate.itertuples(index=False)):
        ax.text(val + 0.01, i, f"{val:.0%}", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Pop, reggaeton, and related mainstream genres have the highest proportion of "
        "popular tracks (over 70%), while niche genres have much lower rates. Genre is "
        "a strong categorical predictor of popularity."
    )

    st.markdown("---")

    # --- Correlation heatmap ---
    st.subheader("Correlation Heatmap")
    corr_cols = AUDIO_FEATURES + ["duration_ms", "is_popular"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f",
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Audio Features")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "Energy and loudness are strongly positively correlated (+0.76), while both are "
        "inversely related to acousticness. These multicollinearity patterns are handled "
        "gracefully by tree-based models through regularization."
    )


# ===================================================================
# TAB 3 — Model Performance
# ===================================================================
with tab3:
    st.header("Model Performance")

    if metrics_df is not None:
        st.subheader("Comparison Table")
        st.dataframe(metrics_df.style.format("{:.4f}").highlight_max(axis=0, color="#d4edda"),
                      use_container_width=True)

        st.markdown("---")

        # Bar chart
        st.subheader("F1 Score Comparison")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
        metrics_df["F1"].plot(kind="barh", ax=ax, color=colors[:len(metrics_df)])
        ax.set_xlabel("F1 Score")
        ax.set_title("Model Comparison — F1 Score")
        for i, v in enumerate(metrics_df["F1"]):
            ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")

        # ROC curves
        st.subheader("ROC Curves")
        roc_path = os.path.join(FIGURES_DIR, "roc_curves_final.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.info("ROC curve image not found. Run the notebook to generate it.")

        st.markdown("---")

        # Best hyperparameters
        st.subheader("Best Hyperparameters")
        for model_name, params in best_params.items():
            with st.expander(f"{model_name}"):
                for k, v in params.items():
                    st.write(f"**{k}:** {v}")
    else:
        st.warning("Metrics summary not found. Please run the notebook first.")


# ===================================================================
# TAB 4 — Explainability & Interactive Prediction
# ===================================================================
with tab4:
    st.header("Explainability & Interactive Prediction")

    # --- SHAP plots ---
    col_shap1, col_shap2 = st.columns(2)
    with col_shap1:
        st.subheader("SHAP Beeswarm Plot")
        beeswarm_path = os.path.join(FIGURES_DIR, "shap_beeswarm.png")
        if os.path.exists(beeswarm_path):
            st.image(beeswarm_path, use_container_width=True)
        else:
            st.info("SHAP beeswarm plot not found.")

    with col_shap2:
        st.subheader("SHAP Feature Importance (Bar)")
        bar_path = os.path.join(FIGURES_DIR, "shap_bar.png")
        if os.path.exists(bar_path):
            st.image(bar_path, use_container_width=True)
        else:
            st.info("SHAP bar plot not found.")

    st.markdown("---")

    # --- Interactive Prediction ---
    st.subheader("Interactive Prediction")

    # Model selector
    available_models = list(sklearn_models.keys())
    if mlp_model is not None:
        available_models.append("MLP")
    selected_model_name = st.selectbox("Select a model", available_models)

    st.markdown("**Adjust the audio features below to predict popularity:**")

    # Feature inputs — use sliders for the most important features
    col1, col2, col3 = st.columns(3)

    with col1:
        danceability = st.slider("Danceability", 0.0, 1.0, 0.6, 0.01)
        energy = st.slider("Energy", 0.0, 1.0, 0.7, 0.01)
        loudness = st.slider("Loudness (dB)", -60.0, 0.0, -6.0, 0.5)

    with col2:
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.1, 0.01)
        valence = st.slider("Valence", 0.0, 1.0, 0.5, 0.01)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05, 0.01)

    with col3:
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0, 0.01)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.15, 0.01)
        tempo = st.slider("Tempo (BPM)", 40.0, 250.0, 120.0, 1.0)

    # Genre selector
    top_genres = train_test_info.get("top_genres", ["pop", "rock", "hip-hop"])
    genre = st.selectbox("Genre", top_genres + ["other"])

    # Use median/mode for features not exposed via sliders
    duration_ms = df["duration_ms"].median()

    # Build input DataFrame
    user_input = pd.DataFrame([{
        "danceability": danceability,
        "energy": energy,
        "loudness": loudness,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
        "duration_ms": duration_ms,
        "track_genre_top": genre,
        "explicit": False,
        "key": 0,
        "mode": 1,
        "time_signature": 4,
    }])

    if st.button("Predict", type="primary"):
        if selected_model_name == "MLP":
            if mlp_model is not None and mlp_preprocessor is not None:
                X_transformed = mlp_preprocessor.transform(user_input)
                prob = float(mlp_model.predict(X_transformed, verbose=0).ravel()[0])
                pred = int(prob >= 0.5)
            else:
                st.error("MLP model not loaded.")
                pred, prob = None, None
        else:
            model = sklearn_models.get(selected_model_name)
            if model is not None:
                pred = int(model.predict(user_input)[0])
                prob = float(model.predict_proba(user_input)[0][1])
            else:
                st.error(f"Model '{selected_model_name}' not loaded.")
                pred, prob = None, None

        if pred is not None:
            # Display prediction
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                label = "Popular" if pred == 1 else "Not Popular"
                color = "green" if pred == 1 else "red"
                st.markdown(
                    f"### Prediction: :{color}[**{label}**]"
                )
                st.metric("Probability of being Popular", f"{prob:.1%}")

            # SHAP waterfall for the prediction
            with col_res2:
                if selected_model_name in ["Random Forest", "XGBoost", "Decision Tree"]:
                    try:
                        model_obj = sklearn_models[selected_model_name]
                        pre_step = model_obj.named_steps["pre"]
                        clf_step = model_obj.named_steps["clf"]

                        feature_cols_num = train_test_info.get("feature_cols_num", NUM_FEATURES)
                        feature_cols_cat = train_test_info.get(
                            "feature_cols_cat",
                            ["track_genre_top", "explicit", "key", "mode", "time_signature"],
                        )
                        feature_names_out = (
                            list(feature_cols_num)
                            + list(pre_step.named_transformers_["cat"]
                                   .get_feature_names_out(feature_cols_cat))
                        )

                        X_transformed = pre_step.transform(user_input)
                        X_df = pd.DataFrame(X_transformed, columns=feature_names_out)

                        explainer = shap.TreeExplainer(clf_step)
                        shap_vals = explainer(X_df)

                        # Handle multi-output (binary) SHAP values — use class 1 (popular)
                        sv = shap_vals[0]
                        if len(sv.shape) > 1 and sv.shape[-1] == 2:
                            sv = sv[..., 1]

                        fig_w, ax_w = plt.subplots(figsize=(8, 5))
                        shap.plots.waterfall(sv, max_display=12, show=False)
                        plt.title("SHAP Waterfall — Your Custom Input")
                        plt.tight_layout()
                        st.pyplot(fig_w)
                        plt.close(fig_w)
                    except Exception as e:
                        st.warning(f"SHAP waterfall could not be generated: {e}")
                else:
                    st.info(
                        "SHAP waterfall plots are available for tree-based models "
                        "(Decision Tree, Random Forest, XGBoost)."
                    )
