from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
PLOTS_DIR    = PROJECT_ROOT / "plots"
RESULTS_DIR  = PROJECT_ROOT / "results"
LOGS_DIR     = PROJECT_ROOT / "logs"
ENV_FILE     = PROJECT_ROOT / ".env"

APP_ENTRYPOINT = SRC_DIR / "app.py"

# ── Streamlit ─────────────────────────────────────────────────────────────────
STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501

# ── Data settings ─────────────────────────────────────────────────────────────
# Leagues covered by soccerdata / FBref
# Codes : ENG-Premier League, ENG-Championship, ESP-La Liga, FRA-Ligue 1,
#         FRA-Ligue 2, GER-Bundesliga, ITA-Serie A, ITA-Serie B,
#         NED-Eredivisie, BEL-First Division A, POR-Primeira Liga
TARGET_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "FRA-Ligue 1",
    "GER-Bundesliga",
    "ITA-Serie A",
]
SEASON = "2324"   # soccerdata season string for 2023-24

# Minimum minutes played to include a player
MIN_MINUTES = 500

# ── Model registry ────────────────────────────────────────────────────────────
MODELS = {
    "kmeans_team": {
        "name": "KMeans — Style d'équipe",
        "description": (
            "Clustering KMeans sur les statistiques collectives des équipes "
            "pour identifier leur style de jeu (possession, pressing, direct...)."
        ),
        "path": MODELS_DIR / "kmeans_team.joblib",
    },
    "rf_role": {
        "name": "Random Forest — Rôle tactique",
        "description": (
            "Classificateur Random Forest qui prédit le rôle tactique réel "
            "d'un joueur (gardien, défenseur, milieu, attaquant) depuis ses stats."
        ),
        "path": MODELS_DIR / "rf_role.joblib",
    },
    "knn_similarity": {
        "name": "KNN — Similarité joueur",
        "description": (
            "K-Nearest Neighbors dans l'espace des features normalisées "
            "pour trouver les joueurs au profil le plus compatible."
        ),
        "path": MODELS_DIR / "knn_similarity.joblib",
    },
}
