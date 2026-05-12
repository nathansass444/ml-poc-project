"""
model_io.py  –  Chargement des modèles depuis le disque.

Supporte : .joblib, .pkl, .pickle
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import joblib


def load_model(path: Path) -> Any:
    """
    Charge un modèle sérialisé depuis le disque.

    Parameters
    ----------
    path : Path
        Chemin vers le fichier modèle (.joblib / .pkl / .pickle).

    Returns
    -------
    Any
        Objet modèle chargé. Peut être un sklearn Pipeline ou un dict
        (cas du knn_similarity qui contient scaler + knn + index).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {path}\n"
            "Lancez d'abord : python scripts/train_models.py"
        )

    suffix = path.suffix.lower()

    if suffix == ".joblib":
        obj = joblib.load(path)
    elif suffix in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            obj = pickle.load(f)
    else:
        raise ValueError(f"Format non supporté : {suffix}. Utilisez .joblib, .pkl ou .pickle")

    # Le template s'attend à un objet avec .predict()
    # Pour les Pipelines sklearn c'est natif.
    # Pour le dict knn_similarity on crée un wrapper léger.
    if isinstance(obj, dict) and "knn" in obj:
        return _KNNWrapper(obj)

    return obj


class _KNNWrapper:
    """
    Wrapper pour le modèle KNN de similarité.
    Expose une méthode .predict() compatible avec le template,
    qui retourne pour chaque joueur son cluster de rôle prédit
    à partir du modèle rf_role (fallback : index du voisin le plus proche).
    """

    def __init__(self, knn_data: dict) -> None:
        self.scaler       = knn_data["scaler"]
        self.knn          = knn_data["knn"]
        self.player_index = knn_data["player_index"]
        self.feature_names = knn_data.get("feature_names", [])

    def predict(self, X: Any) -> Any:
        """
        Retourne l'indice du joueur le plus similaire pour chaque ligne de X.
        Utilisé uniquement par le pipeline d'évaluation du template.
        """
        import numpy as np
        X_arr = X.values if hasattr(X, "values") else X
        X_scaled = self.scaler.transform(X_arr)
        distances, indices = self.knn.kneighbors(X_scaled, n_neighbors=1)
        return indices.flatten()

    def find_similar(self, X_single: Any, n: int = 10) -> "pd.DataFrame":
        """
        Trouve les n joueurs les plus similaires à un profil donné.
        Utilisé par l'application Streamlit.
        """
        import numpy as np
        import pandas as pd

        X_arr = X_single.values if hasattr(X_single, "values") else X_single
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        X_scaled = self.scaler.transform(X_arr)
        distances, indices = self.knn.kneighbors(X_scaled, n_neighbors=n + 1)

        results = self.player_index.iloc[indices[0][1:]].copy()  # exclure lui-même
        results["similarity_score"] = 1 - distances[0][1:]
        return results.reset_index(drop=True)
