"""
results.py  –  Sauvegarde des métriques d'évaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import RESULTS_DIR  # noqa: E402


def write_metrics(rows: list[dict]) -> pd.DataFrame:
    """
    Sauvegarde les métriques dans results/model_metrics.csv et retourne le DataFrame.

    Parameters
    ----------
    rows : list[dict]
        Liste de dictionnaires, un par modèle évalué.

    Returns
    -------
    pd.DataFrame
        DataFrame des métriques.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)

    output_path = RESULTS_DIR / "model_metrics.csv"
    df.to_csv(output_path, index=False)

    return df
