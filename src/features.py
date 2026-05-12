"""
features.py  -  Feature engineering base sur les colonnes reelles de FBref.

Colonnes disponibles verifiees sur players_raw.csv et teams_raw.csv.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Features equipe  —  style de jeu (KMeans)
# ─────────────────────────────────────────────────────────────────────────────

# Colonnes directement disponibles dans teams_raw.csv
TEAM_DIRECT_FEATURES = [
    "Poss",                   # % possession
    "Standard_Sh/90",         # tirs / 90
    "Standard_SoT/90",        # tirs cadres / 90
    "Standard_SoT%",          # precision tirs
    "Standard_G/SoT",         # conversion
    "Per 90 Minutes_Gls",     # buts marques / 90
    "Team Success_PPM",       # points / match
    "Team Success_+/-90",     # difference de buts / 90
]

# Colonnes a calculer en per-90 (raw / Playing Time_90s)
TEAM_COMPUTED_PER90 = {
    "fls_per90":  "Performance_Fls",   # fautes (proxy pressing agressif)
    "int_per90":  "Performance_Int",   # interceptions (bloc defensif)
    "tklw_per90": "Performance_TklW",  # tacles reussis
    "crs_per90":  "Performance_Crs",   # centres (jeu direct/large)
    "off_per90":  "Performance_Off",   # hors-jeu (ligne haute)
    "crd_per90":  "Performance_CrdY",  # cartons jaunes (agressivite)
}


def prepare_team_features(teams_df: pd.DataFrame) -> pd.DataFrame:
    """Extrait et nettoie les features style d equipe pour KMeans."""
    X = pd.DataFrame(index=teams_df.index)

    # Features directes
    for col in TEAM_DIRECT_FEATURES:
        if col in teams_df.columns:
            X[col] = pd.to_numeric(teams_df[col], errors="coerce")

    # Features calculees per-90
    mp_col = "Playing Time_90s" if "Playing Time_90s" in teams_df.columns else "Playing Time_MP"
    if mp_col in teams_df.columns:
        denom = teams_df[mp_col].clip(lower=1)
        for feat_name, raw_col in TEAM_COMPUTED_PER90.items():
            if raw_col in teams_df.columns:
                X[feat_name] = pd.to_numeric(teams_df[raw_col], errors="coerce") / denom

    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    X = X.loc[:, X.std() > 0]
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Features joueur  —  profil tactique (RF + KNN)
# ─────────────────────────────────────────────────────────────────────────────

# Colonnes directement disponibles dans players_raw.csv (deja en per-90)
PLAYER_DIRECT_FEATURES = [
    "Per 90 Minutes_Gls",     # buts / 90
    "Per 90 Minutes_Ast",     # passes decisives / 90
    "Per 90 Minutes_G+A",     # contributions offensives / 90
    "Standard_Sh/90",         # tirs / 90
    "Standard_SoT/90",        # tirs cadres / 90
    "Standard_G/SoT",         # conversion
    "Standard_SoT%",          # precision tirs
]

# Colonnes a calculer en per-90 (raw / Playing Time_90s)
PLAYER_COMPUTED_PER90 = {
    "int_per90":  "Performance_Int",   # interceptions
    "tklw_per90": "Performance_TklW",  # tacles reussis
    "fls_per90":  "Performance_Fls",   # fautes commises (agressivite)
    "fld_per90":  "Performance_Fld",   # fautes subies (dribbles proxy)
    "crs_per90":  "Performance_Crs",   # centres
    "off_per90":  "Performance_Off",   # hors-jeu (attaquants)
    "crd_per90":  "Performance_CrdY",  # cartons jaunes
}


def prepare_player_features(players_df: pd.DataFrame) -> pd.DataFrame:
    """Extrait et nettoie les features joueur pour RF et KNN."""
    X = pd.DataFrame(index=players_df.index)

    # Features directes
    for col in PLAYER_DIRECT_FEATURES:
        if col in players_df.columns:
            X[col] = pd.to_numeric(players_df[col], errors="coerce")

    # Features calculees per-90
    if "Playing Time_90s" in players_df.columns:
        denom = players_df["Playing Time_90s"].clip(lower=1)
        for feat_name, raw_col in PLAYER_COMPUTED_PER90.items():
            if raw_col in players_df.columns:
                X[feat_name] = pd.to_numeric(players_df[raw_col], errors="coerce") / denom

    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    # Supprimer les colonnes constantes (std=0) qui feraient planter le scaler
    X = X.loc[:, X.std() > 0]
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Role tactique  —  mapping position -> label (pour Random Forest)
# ─────────────────────────────────────────────────────────────────────────────

POSITION_TO_ROLE = {
    "GK": "goalkeeper",
    "DF": "defender",
    "MF": "midfielder",
    "FW": "forward",
}


def get_role_label(position_str: str) -> str:
    """Convertit la position FBref (ex: 'MF,FW') en role simplifie."""
    if pd.isna(position_str) or str(position_str).strip() == "":
        return "unknown"
    primary = str(position_str).split(",")[0].strip().upper()
    return POSITION_TO_ROLE.get(primary, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Nommage clusters KMeans
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_STYLE_NAMES = {
    0: "Possession",
    1: "Pressing / intensite",
    2: "Jeu direct / physique",
    3: "Bloc bas / contre-attaque",
}


def label_team_cluster(cluster_id: int) -> str:
    return CLUSTER_STYLE_NAMES.get(cluster_id, f"Style {cluster_id}")
