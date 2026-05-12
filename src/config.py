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

# ── Incremental mode ──────────────────────────────────────────────────────────
# Phase 1 : TEST_MODE = True  → 1 ligue, 1 club
# Phase 2 : TEST_MODE = False → toutes les ligues
TEST_MODE   = True
TEST_LEAGUE = "ENG-Championship"
TEST_CLUB   = "Sheffield United"

# ── Data settings ─────────────────────────────────────────────────────────────
# Ligues secondaires : vivier principal de joueurs à recruter
TARGET_LEAGUES = [
    "ENG-Championship",
    "FRA-Ligue 2",
    "GER-2. Bundesliga",
    "ITA-Serie B",
    "NED-Eredivisie",
    "BEL-First Division A",
    "POR-Primeira Liga",
    "ESP-Segunda División",
]

# Big 5 : joueurs peu utilisés potentiellement disponibles pour les ligues secondaires
# Ajoutés au pipeline final avec filtre MIN_MINUTES_TOP5 pour ne garder que les remplaçants
TOP5_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "FRA-Ligue 1",
    "ITA-Serie A",
]

# Seuil de minutes pour qu'un joueur du top 5 soit considéré "disponible"
# (titulaire = ~2500 min sur 38 matchs, on garde les joueurs peu utilisés)
MIN_MINUTES_TOP5 = 900

SEASON      = "2526"   # saison 2025-26
MIN_MINUTES = 500      # minutes minimum pour inclure un joueur (ligues secondaires)

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
