import streamlit as st
import pandas as pd
import numpy as np

from src.model import load_artifacts, predict_proba

st.set_page_config(page_title="EPL Match Predictor", layout="centered")
st.title("⚽ EPL Match Result Predictor")
st.write("Select teams and get predicted probabilities (Away / Draw / Home).")

# Load UI-ready features
ui_df = pd.read_csv("data/ui_features.csv")
ui_df["MatchDate"] = pd.to_datetime(ui_df["MatchDate"])

teams = sorted(set(ui_df["HomeTeam"]).union(set(ui_df["AwayTeam"])))

home_team = st.selectbox("Home Team", teams)
away_team = st.selectbox("Away Team", teams)

if home_team == away_team:
    st.warning("Home and Away teams must be different.")

# Load trained model artifacts
model, scaler, feature_cols = load_artifacts("artifacts")

def get_latest_features(home_team, away_team):
    # Find the most recent match row for this pairing OR fallback to latest available team forms
    pair_rows = ui_df[(ui_df["HomeTeam"] == home_team) & (ui_df["AwayTeam"] == away_team)]
    if len(pair_rows) > 0:
        row = pair_rows.sort_values("MatchDate").iloc[-1]
    else:
        # fallback: latest home form of home_team + latest away form of away_team
        home_latest = ui_df[ui_df["HomeTeam"] == home_team].sort_values("MatchDate").iloc[-1]
        away_latest = ui_df[ui_df["AwayTeam"] == away_team].sort_values("MatchDate").iloc[-1]

        row = pd.Series({
            "home_gf_form": home_latest["home_gf_form"],
            "home_ga_form": home_latest["home_ga_form"],
            "home_matches": home_latest["home_matches"],
            "away_gf_form": away_latest["away_gf_form"],
            "away_ga_form": away_latest["away_ga_form"],
            "away_matches": away_latest["away_matches"],
        })

    # Build a 1-row dataframe with correct feature order
    X = pd.DataFrame([row[feature_cols].to_dict()]) if isinstance(row, pd.Series) else pd.DataFrame([row[feature_cols]])
    return X

if st.button("Predict"):
    if home_team != away_team:
        X = get_latest_features(home_team, away_team)
        probas = predict_proba(model, scaler, X)[0]  # [Away, Draw, Home]

        pred_idx = int(np.argmax(probas))
        label_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}

        st.subheader(f"Prediction: **{label_map[pred_idx]}**")
        st.write("Probabilities:")
        st.dataframe(pd.DataFrame({
            "Outcome": ["Away Win", "Draw", "Home Win"],
            "Probability": probas
        }))

        st.bar_chart(pd.DataFrame({"Probability": probas}, index=["Away", "Draw", "Home"]))
