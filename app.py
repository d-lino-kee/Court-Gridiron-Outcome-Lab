"""Streamlit demo for the Court & Gridiron Outcome Lab.

Run from project root:  streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = Path("models")

st.set_page_config(page_title="Court & Gridiron Outcome Lab", page_icon="🏀", layout="wide")

st.title("🏀🏈 Court & Gridiron Outcome Lab")
st.caption(
    "Interactive home-team win-probability predictor. "
    "Models were trained on synthetic NBA and NFL game datasets with hyperparameter-tuned "
    "Logistic Regression, Random Forest, and XGBoost pipelines."
)

# ---------- sidebar: sport + model picker ----------
with st.sidebar:
    st.header("Configuration")
    sport = st.radio("Sport", ["NBA", "NFL"], horizontal=True).lower()
    model_label = st.radio("Model", ["XGBoost", "Random Forest", "Logistic Regression"], index=0)
    model_key = {"XGBoost": "xgb", "Random Forest": "rf", "Logistic Regression": "lr"}[model_label]

    model_path = MODELS_DIR / f"{sport}_{model_key}.joblib"
    st.markdown(f"**Model file:** `{model_path}`")
    if not model_path.exists():
        st.error(
            f"Model not found at `{model_path}`.\n\n"
            "Train it first:\n"
            f"```bash\npython src/train.py --input data/simulated_games_{sport}.csv "
            "--models lr rf xgb --cv 5 --scoring f1 --n_iter 25\n```"
        )
        st.stop()

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

bundle = load_model(str(model_path))
pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]


def base_row(sport: str) -> dict:
    """Sensible neutral defaults for every feature the model expects."""
    common = {"season": 2023, "date": "2023-01-01", "game_id": 0}
    if sport == "nba":
        return {
            **common,
            "home_elo": 1520, "away_elo": 1500, "elo_diff": 20,
            "home_off_rating_5": 113, "away_off_rating_5": 112,
            "home_def_rating_5": 111, "away_def_rating_5": 112,
            "home_rest": 2, "away_rest": 2,
            "home_b2b": 0, "away_b2b": 0,
            "home_travel_km": 100, "away_travel_km": 1500,
            "injured_starters_home": 0, "injured_starters_away": 0,
            "vegas_spread": -2.0,
        }
    return {
        **common,
        "home_elo": 1520, "away_elo": 1500, "elo_diff": 20,
        "home_qb_rating": 95, "away_qb_rating": 92, "qb_rating_diff": 3,
        "home_rush_off": 0.1, "away_rush_off": -0.1,
        "home_rush_def": -0.1, "away_rush_def": 0.1,
        "home_pass_off": 0.2, "away_pass_off": -0.2,
        "home_pass_def": -0.1, "away_pass_def": 0.1,
        "home_rest": 7, "away_rest": 7,
        "home_b2b": 0, "away_b2b": 0,
        "home_travel_km": 200, "away_travel_km": 2500,
        "injured_starters_home": 1, "injured_starters_away": 1,
        "weather_index": 0.0,
        "vegas_spread": -2.5,
    }


row = base_row(sport)

# ---------- main UI: feature inputs ----------
st.subheader(f"{sport.upper()} game inputs")
st.caption("Adjust the inputs below. Anything you don't set keeps a neutral default.")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Team strength**")
    row["home_elo"] = st.slider("Home Elo", 1300, 1700, int(row["home_elo"]))
    row["away_elo"] = st.slider("Away Elo", 1300, 1700, int(row["away_elo"]))
    row["elo_diff"] = row["home_elo"] - row["away_elo"]
    if sport == "nfl":
        row["home_qb_rating"] = st.slider("Home QB rating", 50, 140, int(row["home_qb_rating"]))
        row["away_qb_rating"] = st.slider("Away QB rating", 50, 140, int(row["away_qb_rating"]))
        row["qb_rating_diff"] = row["home_qb_rating"] - row["away_qb_rating"]

with c2:
    st.markdown("**Recent form**")
    if sport == "nba":
        row["home_off_rating_5"] = st.slider("Home off rating (last 5)", 95.0, 130.0, float(row["home_off_rating_5"]), 0.5)
        row["away_off_rating_5"] = st.slider("Away off rating (last 5)", 95.0, 130.0, float(row["away_off_rating_5"]), 0.5)
        row["home_def_rating_5"] = st.slider("Home def rating (last 5)", 95.0, 130.0, float(row["home_def_rating_5"]), 0.5)
        row["away_def_rating_5"] = st.slider("Away def rating (last 5)", 95.0, 130.0, float(row["away_def_rating_5"]), 0.5)
    else:
        row["home_pass_off"] = st.slider("Home pass-offense z", -3.0, 3.0, float(row["home_pass_off"]), 0.1)
        row["away_pass_off"] = st.slider("Away pass-offense z", -3.0, 3.0, float(row["away_pass_off"]), 0.1)
        row["home_pass_def"] = st.slider("Home pass-defense z", -3.0, 3.0, float(row["home_pass_def"]), 0.1)
        row["away_pass_def"] = st.slider("Away pass-defense z", -3.0, 3.0, float(row["away_pass_def"]), 0.1)
        row["home_rush_off"] = st.slider("Home rush-offense z", -3.0, 3.0, float(row["home_rush_off"]), 0.1)
        row["away_rush_off"] = st.slider("Away rush-offense z", -3.0, 3.0, float(row["away_rush_off"]), 0.1)

with c3:
    st.markdown("**Context**")
    if sport == "nba":
        row["home_rest"] = st.slider("Home rest days", 0, 4, int(row["home_rest"]))
        row["away_rest"] = st.slider("Away rest days", 0, 4, int(row["away_rest"]))
    else:
        row["home_rest"] = st.slider("Home rest days", 3, 14, int(row["home_rest"]))
        row["away_rest"] = st.slider("Away rest days", 3, 14, int(row["away_rest"]))
    row["home_b2b"] = int(row["home_rest"] == 0) if sport == "nba" else int(row["home_rest"] <= 4)
    row["away_b2b"] = int(row["away_rest"] == 0) if sport == "nba" else int(row["away_rest"] <= 4)
    row["injured_starters_home"] = st.slider("Home injured starters", 0, 5, int(row["injured_starters_home"]))
    row["injured_starters_away"] = st.slider("Away injured starters", 0, 5, int(row["injured_starters_away"]))
    row["away_travel_km"] = st.slider("Away team travel (km)", 0, 6000, int(row["away_travel_km"]), 100)
    row["vegas_spread"] = st.slider("Vegas spread (home favored if negative)", -20.0, 20.0, float(row["vegas_spread"]), 0.5)
    if sport == "nfl":
        row["weather_index"] = st.slider("Weather index (-1 bad → 1 good)", -1.0, 1.0, float(row["weather_index"]), 0.05)

# ---------- predict ----------
# Build a one-row dataframe matching the columns the pipeline was trained on.
X_one = pd.DataFrame([{c: row.get(c, 0) for c in feature_columns}])
prob_home = float(pipeline.predict_proba(X_one)[0, 1])
prob_away = 1.0 - prob_home

st.divider()
left, right = st.columns([2, 3])
with left:
    st.metric("Home win probability", f"{prob_home*100:.1f}%")
    st.metric("Away win probability", f"{prob_away*100:.1f}%")
    pick = "HOME" if prob_home >= 0.5 else "AWAY"
    confidence = max(prob_home, prob_away)
    st.success(f"Model pick: **{pick}**  (confidence {confidence*100:.1f}%)")

with right:
    st.markdown("**Probability bar**")
    st.progress(prob_home, text=f"Home {prob_home*100:.1f}%   |   Away {prob_away*100:.1f}%")
    st.markdown("**Key inputs being scored**")
    show_cols = [c for c in feature_columns if c not in ("season", "date", "game_id")]
    st.dataframe(X_one[show_cols].T.rename(columns={0: "value"}), use_container_width=True, height=320)

with st.expander("How this works"):
    st.markdown(
        """
        The selected model is a scikit-learn `Pipeline` with two steps:

        1. **Preprocessing** — median-impute + standard-scale numeric features, mode-impute + one-hot
           encode categorical ones (`src/features.py`).
        2. **Classifier** — Logistic Regression, Random Forest, or XGBoost, selected here.

        Best hyperparameters were found via stratified 5-fold cross-validated search
        (`GridSearchCV` for LR, `RandomizedSearchCV` for RF / XGB) optimizing F1.
        Models were refit on the full training split and evaluated on a held-out 20% test set.

        The probability above is `pipeline.predict_proba(...)` on a single synthetic game
        constructed from the inputs you set.
        """
    )
