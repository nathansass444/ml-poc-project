"""
data.py  –  Implémentation de load_dataset_split() pour le template.

Le template appelle cette fonction pour obtenir (X_train, X_test, y_train, y_test).
Dans notre projet, on évalue le classificateur de rôle tactique (rf_role),
donc on retourne un split du dataset joueurs avec les labels de rôle.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

# Permettre l'import depuis scripts/main.py
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import DATA_DIR, MIN_MINUTES  # noqa: E402
from features import prepare_player_features, get_role_label  # noqa: E402


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """
    Charge le dataset joueurs et retourne un split train/test.

    Returns
    -------
    X_train, X_test, y_train, y_test
        X : DataFrame de features tactiques (voir features.PLAYER_ROLE_FEATURES)
        y : Series de rôles tactiques (goalkeeper, defender, central_mid, etc.)
    """
    players_path = DATA_DIR / "players_raw.csv"

    if not players_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {players_path}\n"
            "Lancez d'abord : python scripts/collect_data.py"
        )

    df = pd.read_csv(players_path)

    # Filtrer les joueurs avec assez de minutes
    minutes_col = next(
        (c for c in df.columns if c.lower() in ["minutes", "min", "mins", "minutes_90s"]),
        None,
    )
    if minutes_col:
        df = df[df[minutes_col] >= MIN_MINUTES].copy()

    # Labels
    df["role_label"] = df["position"].apply(get_role_label)
    df = df[df["role_label"] != "unknown"].copy()

    if len(df) < 50:
        raise ValueError(
            f"Trop peu de joueurs après filtrage ({len(df)}). "
            "Vérifiez que collect_data.py a bien tourné."
        )

    X = prepare_player_features(df)
    y = df["role_label"].reset_index(drop=True)
    X = X.reset_index(drop=True)

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
