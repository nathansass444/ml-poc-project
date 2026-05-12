"""
metrics.py  –  Implémentation de compute_metrics() pour le template.

Évalue la qualité du classificateur de rôle tactique.
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """
    Calcule les métriques de classification du rôle tactique.

    Parameters
    ----------
    y_true : array-like
        Rôles réels (labels de position FBref).
    y_pred : array-like
        Rôles prédits par le modèle.

    Returns
    -------
    dict[str, float]
        Dictionnaire de métriques numériques.
    """
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_weighted":       float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "precision_weighted":float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted":   float(recall_score(y_true, y_pred, average="weighted",    zero_division=0)),
    }
