"""
train_models.py  -  Entraine et sauvegarde les 3 modeles du projet.

    python scripts/train_models.py

A executer APRES collect_data.py.
Les modeles sont sauvegardes dans models/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import DATA_DIR, MODELS_DIR  # noqa: E402
from features import (                   # noqa: E402
    get_role_label,
    prepare_team_features,
    prepare_player_features,
)


def train_kmeans_team(teams_df: pd.DataFrame) -> Pipeline:
    print("KMeans — style equipe...")
    X = prepare_team_features(teams_df)
    print(f"  {X.shape[1]} features : {list(X.columns)}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=4, random_state=42, n_init=20)),
    ])
    pipeline.fit(X)

    labels = pipeline.named_steps["kmeans"].labels_
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Distribution clusters : {dict(zip(unique.tolist(), counts.tolist()))}")
    return pipeline


def train_rf_role(players_df: pd.DataFrame) -> Pipeline:
    print("Random Forest — role tactique...")
    df = players_df.copy()

    # La colonne position s'appelle "pos" dans FBref via soccerdata
    pos_col = "pos" if "pos" in df.columns else "position"
    df["role_label"] = df[pos_col].apply(get_role_label)
    df = df[df["role_label"] != "unknown"].copy()

    X = prepare_player_features(df)
    y = df["role_label"]

    print(f"  {len(df):,} joueurs | {X.shape[1]} features | classes : {sorted(y.unique())}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X, y)
    return pipeline


def train_knn_similarity(players_df: pd.DataFrame) -> dict:
    print("KNN — similarite joueur...")
    X = prepare_player_features(players_df)
    print(f"  {len(players_df):,} joueurs | {X.shape[1]} features")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = NearestNeighbors(n_neighbors=11, metric="cosine", n_jobs=-1)
    knn.fit(X_scaled)

    pos_col = "pos" if "pos" in players_df.columns else "position"
    player_index = players_df[["player", "team", "league", pos_col, "tier"]].reset_index(drop=True)

    return {
        "scaler":        scaler,
        "knn":           knn,
        "player_index":  player_index,
        "feature_names": list(X.columns),
        "X_scaled":      X_scaled,   # garde les vecteurs pour la recherche
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    players_path = DATA_DIR / "players_raw.csv"
    teams_path   = DATA_DIR / "teams_raw.csv"

    if not players_path.exists() or not teams_path.exists():
        print("Donnees manquantes. Lancez d abord : python scripts/collect_data.py")
        sys.exit(1)

    players_df = pd.read_csv(players_path)
    teams_df   = pd.read_csv(teams_path)

    print(f"{len(players_df):,} joueurs charges")
    print(f"{len(teams_df):,} equipes chargees\n")

    kmeans   = train_kmeans_team(teams_df)
    rf       = train_rf_role(players_df)
    knn_data = train_knn_similarity(players_df)

    joblib.dump(kmeans,   MODELS_DIR / "kmeans_team.joblib")
    joblib.dump(rf,       MODELS_DIR / "rf_role.joblib")
    joblib.dump(knn_data, MODELS_DIR / "knn_similarity.joblib")

    print("\nModeles sauvegardes dans models/")
    print("  kmeans_team.joblib")
    print("  rf_role.joblib")
    print("  knn_similarity.joblib")


if __name__ == "__main__":
    main()
