"""
collect_data.py  —  Télécharge les stats joueurs et équipes depuis FBref.

    python scripts/collect_data.py

Génère data/players_raw.csv et data/teams_raw.csv.
En TEST_MODE (config.py) : 1 ligue, 1 club seulement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import (  # noqa: E402
    DATA_DIR,
    MIN_MINUTES,
    SEASON,
    TARGET_LEAGUES,
    TEST_CLUB,
    TEST_LEAGUE,
    TEST_MODE,
)

try:
    import soccerdata as sd
except ImportError:
    print("soccerdata non installé. Lancez : pip install soccerdata")
    sys.exit(1)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(c) for c in col).strip("_") for col in df.columns]
    return df


def fetch_player_stats(leagues: list[str], club_filter: str | None = None) -> pd.DataFrame:
    print(f"Téléchargement stats joueurs — ligues : {leagues}")
    fbref = sd.FBref(leagues=leagues, seasons=SEASON)

    stat_types = ["standard", "shooting", "playing_time", "misc"]
    frames: list[pd.DataFrame] = []

    for stat in stat_types:
        print(f"  stat_type={stat}...")
        try:
            df = fbref.read_player_season_stats(stat_type=stat)
            df = _flatten_columns(df).reset_index()
            frames.append(df)
        except Exception as exc:
            print(f"  SKIP {stat} ({exc})")

    if not frames:
        raise RuntimeError("Aucune stat joueur récupérée.")

    # Colonnes d'identification présentes dans chaque frame
    id_cols = ["player", "team", "league", "season"]

    merged = frames[0]
    for other in frames[1:]:
        keys = [c for c in id_cols if c in merged.columns and c in other.columns]
        merged = merged.merge(other, on=keys, how="left", suffixes=("", "_dup"))
        merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")], inplace=True)

    # Filtre minutes minimum — cherche "Playing Time_Min" en priorité
    min_col = next(
        (c for c in merged.columns if c.lower() == "playing time_min"),
        next((c for c in merged.columns if c.lower().endswith("_min") and "playing" in c.lower()), None),
    )
    if min_col:
        merged = merged[merged[min_col] >= MIN_MINUTES].copy()
    else:
        print("  WARN: colonne minutes non trouvée, filtre MIN_MINUTES ignoré")

    # Filtre optionnel sur un club
    if club_filter:
        team_col = next((c for c in merged.columns if c.lower() == "team"), None)
        if team_col:
            merged = merged[merged[team_col] == club_filter].copy()
            print(f"  Filtre club : {club_filter} -> {len(merged)} joueurs")

    return merged.reset_index(drop=True)


def fetch_team_stats(leagues: list[str]) -> pd.DataFrame:
    print(f"Téléchargement stats équipes — ligues : {leagues}")
    fbref = sd.FBref(leagues=leagues, seasons=SEASON)

    stat_types = ["standard", "shooting", "playing_time", "misc"]
    frames: list[pd.DataFrame] = []

    for stat in stat_types:
        print(f"  stat_type={stat}...")
        try:
            df = fbref.read_team_season_stats(stat_type=stat)
            df = _flatten_columns(df).reset_index()
            frames.append(df)
        except Exception as exc:
            print(f"  SKIP {stat} ({exc})")

    if not frames:
        raise RuntimeError("Aucune stat équipe récupérée.")

    id_cols = ["team", "league", "season"]

    merged = frames[0]
    for other in frames[1:]:
        keys = [c for c in id_cols if c in merged.columns and c in other.columns]
        merged = merged.merge(other, on=keys, how="left", suffixes=("", "_dup"))
        merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")], inplace=True)

    return merged.reset_index(drop=True)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TEST_MODE:
        print("MODE TEST : 1 ligue, 1 club")
        leagues     = [TEST_LEAGUE]
        club_filter = TEST_CLUB
    else:
        print("MODE COMPLET : toutes les ligues")
        leagues     = TARGET_LEAGUES
        club_filter = None

    players = fetch_player_stats(leagues, club_filter)
    teams   = fetch_team_stats(leagues)

    players_path = DATA_DIR / "players_raw.csv"
    teams_path   = DATA_DIR / "teams_raw.csv"

    players.to_csv(players_path, index=False)
    teams.to_csv(teams_path,     index=False)

    print(f"\n{len(players):,} joueurs  -> {players_path}")
    print(f"{len(teams):,} equipes  -> {teams_path}")
    print("\nColonnes joueurs :", list(players.columns[:10]))
    print("Colonnes equipes :", list(teams.columns[:10]))


if __name__ == "__main__":
    main()
