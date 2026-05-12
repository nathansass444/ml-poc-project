"""
train_models.py  –  Entraîne et sauvegarde les 3 modèles du projet.

    python scripts/train_models.py

À exécuter APRÈS collect_data.py.
Les modèles sont sauvegardés dans models/.
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
    TEAM_STYLE_FEATURES,
    PLAYER_ROLE_FEATURES,
    get_role_label,
    prepare_team_features,
    prepare_player_features,
)


def train_kmeans_team(teams_df: pd.DataFrame) -> Pipeline:
    """
    Clustering KMeans sur le style de jeu des équipes.
    On cherche 4 clusters : possession, pressing, direct, équilibré.
    """
    print("⚙️   Entraînement KMeans — style d'équipe …")
    X = prepare_team_features(teams_df)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=4, random_state=42, n_init=20)),
    ])
    pipeline.fit(X)

    # Nommer les clusters selon le centroïde dominant
    labels = pipeline.named_steps["kmeans"].labels_
    print(f"   Distribution clusters : {dict(zip(*np.unique(labels, return_counts=True)))}")
    return pipeline


def train_rf_role(players_df: pd.DataFrame) -> Pipeline:
    """
    Classification du rôle tactique d'un joueur.
    Le label est dérivé de la colonne 'position' du dataset.
    """
    print("⚙️   Entraînement Random Forest — rôle tactique …")
    df = players_df.copy()
    df["role_label"] = df["position"].apply(get_role_label)

    # Garder uniquement les lignes avec un rôle connu
    df = df[df["role_label"] != "unknown"].copy()

    X = prepare_player_features(df)
    y = df["role_label"]

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

    classes = pipeline.named_steps["rf"].classes_
    print(f"   Classes : {list(classes)}")
    return pipeline


def train_knn_similarity(players_df: pd.DataFrame) -> dict:
    """
    KNN pour la recherche de joueurs similaires.
    On sauvegarde le modèle + le DataFrame indexé pour pouvoir retrouver
    les noms des joueurs à partir des indices KNN.
    """
    print("⚙️   Entraînement KNN — similarité joueur …")
    X = prepare_player_features(players_df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    knn = NearestNeighbors(n_neighbors=11, metric="cosine", n_jobs=-1)
    knn.fit(X_scaled)

    # Index des joueurs pour retrouver noms/équipes après recherche
    player_index = players_df[["player", "team", "league", "position"]].reset_index(drop=True)

    return {
        "scaler": scaler,
        "knn": knn,
        "player_index": player_index,
        "feature_names": list(X.columns),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    players_path = DATA_DIR / "players_raw.csv"
    teams_path   = DATA_DIR / "teams_raw.csv"

    if not players_path.exists() or not teams_path.exists():
        print("❌  Données manquantes. Lancez d'abord : python scripts/collect_data.py")
        sys.exit(1)

    players_df = pd.read_csv(players_path)
    teams_df   = pd.read_csv(teams_path)

    print(f"📂  {len(players_df):,} joueurs chargés")
    print(f"📂  {len(teams_df):,}  équipes chargées\n")

    # ── Train ──
    kmeans   = train_kmeans_team(teams_df)
    rf       = train_rf_role(players_df)
    knn_data = train_knn_similarity(players_df)

    # ── Save ──
    joblib.dump(kmeans,   MODELS_DIR / "kmeans_team.joblib")
    joblib.dump(rf,       MODELS_DIR / "rf_role.joblib")
    joblib.dump(knn_data, MODELS_DIR / "knn_similarity.joblib")

    print("\n✅  Modèles sauvegardés dans models/")
    print("   • kmeans_team.joblib")
    print("   • rf_role.joblib")
    print("   • knn_similarity.joblib")
    print("\n▶️   Tu peux maintenant lancer : python scripts/main.py")


if __name__ == "__main__":
    main()
