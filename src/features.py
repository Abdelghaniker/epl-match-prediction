from __future__ import annotations

import pandas as pd
import numpy as np


FEATURE_COLS = [
    "home_gf_form", "home_ga_form", "home_matches",
    "away_gf_form", "away_ga_form", "away_matches",
]


def load_raw_epl(csv_path: str) -> pd.DataFrame:
    """
    Load raw EPL dataset and apply minimal cleaning + sorting.
    """
    df = pd.read_csv(csv_path)

    df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")
    df = df.dropna(subset=["MatchDate"]).sort_values("MatchDate").reset_index(drop=True)

    # Target encoding: Away=0, Draw=1, Home=2
    df["target"] = df["FullTimeResult"].map({"A": 0, "D": 1, "H": 2})

    return df


def build_ui_features(
    raw_csv_path: str,
    window: int = 5,
    out_csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Build a UI-ready features dataframe that includes:
    MatchDate, HomeTeam, AwayTeam, FEATURE_COLS, target

    Features are leakage-free (shifted rolling window per team).
    """
    df = load_raw_epl(raw_csv_path)

    # Create team-level long format
    home_df = df[[
        "MatchDate", "Season",
        "HomeTeam", "AwayTeam",
        "FullTimeHomeGoals", "FullTimeAwayGoals"
    ]].copy()
    home_df.columns = ["MatchDate", "Season", "Team", "Opponent", "GoalsFor", "GoalsAgainst"]
    home_df["is_home"] = 1

    away_df = df[[
        "MatchDate", "Season",
        "AwayTeam", "HomeTeam",
        "FullTimeAwayGoals", "FullTimeHomeGoals"
    ]].copy()
    away_df.columns = ["MatchDate", "Season", "Team", "Opponent", "GoalsFor", "GoalsAgainst"]
    away_df["is_home"] = 0

    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df = team_df.sort_values(["Team", "MatchDate"]).reset_index(drop=True)

    # Rolling form (no leakage thanks to shift(1))
    gf = team_df.groupby("Team")["GoalsFor"]
    ga = team_df.groupby("Team")["GoalsAgainst"]

    team_df["gf_form"] = (
        gf.shift(1)
          .rolling(window, min_periods=window)
          .mean()
          .reset_index(level=0, drop=True)
    )
    team_df["ga_form"] = (
        ga.shift(1)
          .rolling(window, min_periods=window)
          .mean()
          .reset_index(level=0, drop=True)
    )

    # experience count
    team_df["matches_played"] = team_df.groupby("Team").cumcount()

    # Split back into home/away features
    home_feat = team_df[team_df["is_home"] == 1][
        ["MatchDate", "Team", "gf_form", "ga_form", "matches_played"]
    ].copy()
    away_feat = team_df[team_df["is_home"] == 0][
        ["MatchDate", "Team", "gf_form", "ga_form", "matches_played"]
    ].copy()

    home_feat.columns = ["MatchDate", "HomeTeam", "home_gf_form", "home_ga_form", "home_matches"]
    away_feat.columns = ["MatchDate", "AwayTeam", "away_gf_form", "away_ga_form", "away_matches"]

    # Merge back to match-level
    final_df = df.merge(home_feat, on=["MatchDate", "HomeTeam"], how="left")
    final_df = final_df.merge(away_feat, on=["MatchDate", "AwayTeam"], how="left")

    ui_df = final_df[[
        "MatchDate", "HomeTeam", "AwayTeam",
        *FEATURE_COLS,
        "target"
    ]].dropna().sort_values("MatchDate").reset_index(drop=True)

    if out_csv_path:
        ui_df.to_csv(out_csv_path, index=False)

    return ui_df


def build_model_features(
    raw_csv_path: str,
    window: int = 5,
    out_csv_path: str | None = None,
) -> pd.DataFrame:
    """
    Build the modeling dataframe WITHOUT team names (clean for modeling pipelines):
    MatchDate, FEATURE_COLS, target
    """
    ui_df = build_ui_features(raw_csv_path, window=window, out_csv_path=None)

    model_df = ui_df[[
        "MatchDate",
        *FEATURE_COLS,
        "target"
    ]].copy()

    if out_csv_path:
        model_df.to_csv(out_csv_path, index=False)

    return model_df


def get_latest_feature_row(ui_df: pd.DataFrame, home_team: str, away_team: str) -> pd.DataFrame:
    """
    For an interface: return a 1-row dataframe of feature columns for prediction.

    Strategy:
    1) If there is historical match data for that exact pairing, take the most recent one.
    2) Otherwise, fallback to:
       - latest HOME features available for home_team
       - latest AWAY features available for away_team
    """
    ui_df = ui_df.copy()
    ui_df["MatchDate"] = pd.to_datetime(ui_df["MatchDate"], errors="coerce")
    ui_df = ui_df.dropna(subset=["MatchDate"]).sort_values("MatchDate")

    # 1) exact pairing
    pair = ui_df[(ui_df["HomeTeam"] == home_team) & (ui_df["AwayTeam"] == away_team)]
    if len(pair) > 0:
        row = pair.iloc[-1]
        return pd.DataFrame([row[FEATURE_COLS].to_dict()])

    # 2) fallback
    home_rows = ui_df[ui_df["HomeTeam"] == home_team]
    away_rows = ui_df[ui_df["AwayTeam"] == away_team]

    if len(home_rows) == 0:
        raise ValueError(f"No home matches found for team: {home_team}")
    if len(away_rows) == 0:
        raise ValueError(f"No away matches found for team: {away_team}")

    home_last = home_rows.iloc[-1]
    away_last = away_rows.iloc[-1]

    feat = {
        "home_gf_form": float(home_last["home_gf_form"]),
        "home_ga_form": float(home_last["home_ga_form"]),
        "home_matches": float(home_last["home_matches"]),
        "away_gf_form": float(away_last["away_gf_form"]),
        "away_ga_form": float(away_last["away_ga_form"]),
        "away_matches": float(away_last["away_matches"]),
    }

    return pd.DataFrame([feat])
