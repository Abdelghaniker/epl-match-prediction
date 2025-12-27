import json
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def time_split(df: pd.DataFrame, frac: float = 0.8):
    df = df.sort_values("MatchDate").reset_index(drop=True)
    split_idx = int(len(df) * frac)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train = train_df.drop(columns=["MatchDate", "target"])
    y_train = train_df["target"].astype(int)

    X_test = test_df.drop(columns=["MatchDate", "target"])
    y_test = test_df["target"].astype(int)

    return train_df, test_df, X_train, y_train, X_test, y_test


def train_logreg(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # compatible with older sklearn
    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
        solver="lbfgs"
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def eval_logreg(model, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=["Away", "Draw", "Home"])
    cm = confusion_matrix(y_test, preds)
    return acc, report, cm


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def eval_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=["Away", "Draw", "Home"])
    cm = confusion_matrix(y_test, preds)
    return acc, report, cm


def save_artifacts(model, scaler, feature_cols, out_dir="artifacts"):
    import os
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(model, f"{out_dir}/model.pkl")
    joblib.dump(scaler, f"{out_dir}/scaler.pkl")
    with open(f"{out_dir}/features.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    print(f"✅ Saved artifacts to {out_dir}/")


def load_artifacts(in_dir="artifacts"):
    model = joblib.load(f"{in_dir}/model.pkl")
    scaler = joblib.load(f"{in_dir}/scaler.pkl")
    with open(f"{in_dir}/features.json", "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    return model, scaler, feature_cols


def predict_proba(model, scaler, X_row: pd.DataFrame):
    """
    X_row: dataframe with exactly the feature columns (1 row or many).
    Returns probabilities in order [Away, Draw, Home].
    """
    X_scaled = scaler.transform(X_row)
    return model.predict_proba(X_scaled)
