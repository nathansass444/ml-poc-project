"""
features.py  –  Feature engineering partagé entre l'entraînement et l'inférence.

Toutes les décisions sur QUELLES features utiliser sont centralisées ici
pour garantir la cohérence entre train_models.py, data.py, et app.py.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Features équipe — définir le STYLE DE JEU
# ─────────────────────────────────────────────────────────────────────────────

# Ces features décrivent comment une équipe joue collectivement.
# Noms génériques : soccerdata/FBref peut varier légèrement selon la version.
# La fonction prepare_team_features() gère les aliases.

TEAM_STYLE_FEATURES = [
    # Possession
    "possession",            # % possession moyenne
    # Passes
    "passes_completed_pct",  # précision des passes
    "passes_short_pct",      # % passes courtes (jeu de possession vs direct)
    "passes_progressive",    # passes progressives / 90
    # Pressing
    "pressures",             # nb de pressions défensives / 90
    "press_success_pct",     # % succès des pressings
    "ppda",                  # passes autorisées par action défensive (pressing haut = faible)
    # Transitions
    "shots",                 # tirs / 90 (agressivité offensive)
    "shots_on_target_pct",   # efficacité
    # Duels
    "aerials_won_pct",       # % duels aériens gagnés (jeu direct = élevé)
    # Défense
    "tackles",               # tacles / 90
    "interceptions",         # interceptions / 90
    # Buts
    "goals",                 # buts marqués / 90
    "goals_against",         # buts encaissés / 90
]

# Mapping vers les noms réels que FBref peut utiliser
TEAM_FEATURE_ALIASES: dict[str, list[str]] = {
    "possession":            ["possession", "poss", "Poss"],
    "passes_completed_pct":  ["passes_completed_pct", "cmp_pct", "pass_cmp_pct"],
    "passes_short_pct":      ["passes_short_pct", "short_cmp_pct"],
    "passes_progressive":    ["passes_progressive", "prg_pass", "progressive_passes"],
    "pressures":             ["pressures", "press"],
    "press_success_pct":     ["press_success_pct", "press_succ_pct"],
    "ppda":                  ["ppda", "PPDA"],
    "shots":                 ["shots", "sh", "Sh"],
    "shots_on_target_pct":   ["shots_on_target_pct", "sot_pct", "SoT_pct"],
    "aerials_won_pct":       ["aerials_won_pct", "won_pct_aerial"],
    "tackles":               ["tackles", "tkl", "Tkl"],
    "interceptions":         ["interceptions", "int", "Int"],
    "goals":                 ["goals", "gls", "Gls"],
    "goals_against":         ["goals_against", "ga", "GA"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Features joueur — définir le PROFIL TACTIQUE
# ─────────────────────────────────────────────────────────────────────────────

PLAYER_ROLE_FEATURES = [
    # Volume offensif
    "goals_per90",
    "assists_per90",
    "shots_per90",
    "shots_on_target_per90",
    "xg_per90",
    # Création
    "key_passes_per90",
    "passes_progressive_per90",
    "passes_into_final_third_per90",
    # Pressing / défense
    "pressures_per90",
    "tackles_per90",
    "interceptions_per90",
    "blocks_per90",
    # Dribbles / progression
    "dribbles_completed_per90",
    "carries_progressive_per90",
    # Duels
    "aerials_won_per90",
    "aerials_won_pct",
]

PLAYER_FEATURE_ALIASES: dict[str, list[str]] = {
    "goals_per90":                  ["goals_per90", "gls_per90", "Gls_per90"],
    "assists_per90":                ["assists_per90", "ast_per90"],
    "shots_per90":                  ["shots_per90", "sh_per90"],
    "shots_on_target_per90":        ["shots_on_target_per90", "sot_per90"],
    "xg_per90":                     ["xg_per90", "xG_per90", "npxg_per90"],
    "key_passes_per90":             ["key_passes_per90", "kp_per90"],
    "passes_progressive_per90":     ["passes_progressive_per90", "prg_pass_per90"],
    "passes_into_final_third_per90":["passes_into_final_third_per90", "f3_per90"],
    "pressures_per90":              ["pressures_per90", "press_per90"],
    "tackles_per90":                ["tackles_per90", "tkl_per90"],
    "interceptions_per90":          ["interceptions_per90", "int_per90"],
    "blocks_per90":                 ["blocks_per90", "blk_per90"],
    "dribbles_completed_per90":     ["dribbles_completed_per90", "succ_drb_per90"],
    "carries_progressive_per90":    ["carries_progressive_per90", "prg_carry_per90"],
    "aerials_won_per90":            ["aerials_won_per90", "won_aerial_per90"],
    "aerials_won_pct":              ["aerials_won_pct", "won_pct_aerial"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_columns(df: pd.DataFrame, aliases: dict[str, list[str]]) -> dict[str, str]:
    """Retourne un mapping {feature_name: actual_column} pour les features disponibles."""
    resolved = {}
    for feat, candidates in aliases.items():
        for cand in candidates:
            if cand in df.columns:
                resolved[feat] = cand
                break
    return resolved


def _normalize_per90(df: pd.DataFrame, raw_col: str, minutes_col: str) -> pd.Series:
    """Calcule les stats per90 à la volée si elles n'existent pas."""
    return (df[raw_col] / df[minutes_col].clip(lower=1)) * 90


def prepare_team_features(teams_df: pd.DataFrame) -> pd.DataFrame:
    """Extrait et nettoie les features du style de jeu des équipes."""
    resolved = _resolve_columns(teams_df, TEAM_FEATURE_ALIASES)

    X = pd.DataFrame(index=teams_df.index)
    for feat, col in resolved.items():
        X[feat] = pd.to_numeric(teams_df[col], errors="coerce")

    # Remplir les NaN par la médiane
    X = X.fillna(X.median(numeric_only=True))
    return X


def prepare_player_features(players_df: pd.DataFrame) -> pd.DataFrame:
    """Extrait et nettoie les features du profil tactique des joueurs."""
    resolved = _resolve_columns(players_df, PLAYER_FEATURE_ALIASES)

    # Essayer de calculer les per90 manquants si minutes disponible
    minutes_col = next(
        (c for c in players_df.columns if c.lower() in ["minutes", "min", "mins", "minutes_90s"]),
        None,
    )

    X = pd.DataFrame(index=players_df.index)
    for feat, col in resolved.items():
        X[feat] = pd.to_numeric(players_df[col], errors="coerce")

    # Si certaines per90 manquent, essayer de les dériver
    if minutes_col:
        for feat in PLAYER_ROLE_FEATURES:
            if feat not in X.columns or X[feat].isna().all():
                base = feat.replace("_per90", "")
                if base in players_df.columns:
                    X[feat] = _normalize_per90(players_df, base, minutes_col)

    X = X.fillna(X.median(numeric_only=True))
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Rôle tactique — mapping position → label
# ─────────────────────────────────────────────────────────────────────────────

POSITION_TO_ROLE = {
    # Gardiens
    "GK": "goalkeeper",
    # Défenseurs
    "CB": "defender", "RB": "defender", "LB": "defender",
    "RWB": "defender", "LWB": "defender", "DF": "defender",
    # Milieux défensifs / récupérateurs
    "DM": "defensive_mid", "CDM": "defensive_mid",
    # Milieux centraux / box-to-box
    "CM": "central_mid", "MF": "central_mid",
    # Milieux offensifs / créateurs
    "AM": "attacking_mid", "CAM": "attacking_mid",
    # Ailiers
    "RW": "winger", "LW": "winger", "RM": "winger", "LM": "winger",
    # Attaquants
    "CF": "striker", "ST": "striker", "FW": "striker",
}


def get_role_label(position_str: str) -> str:
    """
    Convertit la chaîne de position FBref (ex: 'MF,FW' ou 'DF') en rôle tactique.
    En cas de position mixte, on prend la première.
    """
    if pd.isna(position_str) or position_str == "":
        return "unknown"
    primary = str(position_str).split(",")[0].strip().upper()
    return POSITION_TO_ROLE.get(primary, "unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Style d'équipe — nommage des clusters KMeans
# ─────────────────────────────────────────────────────────────────────────────

CLUSTER_STYLE_NAMES = {
    0: "Jeu de possession",
    1: "Pressing haut",
    2: "Jeu direct / physique",
    3: "Bloc bas / contre-attaque",
}


def label_team_cluster(cluster_id: int) -> str:
    return CLUSTER_STYLE_NAMES.get(cluster_id, f"Style {cluster_id}")
