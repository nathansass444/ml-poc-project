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
    MIN_MINUTES_TOP5,
    SEASON,
    TARGET_LEAGUES,
    TEST_CLUB,
    TEST_LEAGUE,
    TEST_MODE,
    TOP5_LEAGUES,
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


def fetch_player_stats(
    leagues: list[str],
    min_minutes: int = MIN_MINUTES,
    club_filter: str | None = None,
    tier: str = "secondary",
) -> pd.DataFrame:
    print(f"Fetch joueurs [{tier}] — {leagues}")
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
        raise RuntimeError(f"Aucune stat joueur recuperee pour {leagues}.")

    id_cols = ["player", "team", "league", "season"]

    merged = frames[0]
    for other in frames[1:]:
        keys = [c for c in id_cols if c in merged.columns and c in other.columns]
        merged = merged.merge(other, on=keys, how="left", suffixes=("", "_dup"))
        merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")], inplace=True)

    # Filtre minutes — cherche "Playing Time_Min" en priorite
    min_col = next(
        (c for c in merged.columns if c.lower() == "playing time_min"),
        next((c for c in merged.columns if c.lower().endswith("_min") and "playing" in c.lower()), None),
    )
    if min_col:
        before = len(merged)
        merged = merged[merged[min_col] >= min_minutes].copy()
        print(f"  Filtre {min_minutes} min : {before} -> {len(merged)} joueurs")
    else:
        print("  WARN: colonne minutes non trouvee")

    # Filtre optionnel club
    if club_filter:
        team_col = next((c for c in merged.columns if c.lower() == "team"), None)
        if team_col:
            merged = merged[merged[team_col] == club_filter].copy()
            print(f"  Filtre club : {club_filter} -> {len(merged)} joueurs")

    merged["tier"] = tier
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
        players = fetch_player_stats([TEST_LEAGUE], club_filter=TEST_CLUB, tier="secondary")
        teams   = fetch_team_stats([TEST_LEAGUE])
    else:
        print("MODE COMPLET : ligues secondaires + Top 5 (remplacants)")

        # Ligues secondaires : vivier principal
        players_secondary = fetch_player_stats(
            TARGET_LEAGUES, min_minutes=MIN_MINUTES, tier="secondary"
        )

        # Top 5 : seulement joueurs peu utilises (remplacants potentiellement libérables)
        players_top5 = fetch_player_stats(
            TOP5_LEAGUES, min_minutes=MIN_MINUTES_TOP5, tier="top5"
        )

        players = pd.concat([players_secondary, players_top5], ignore_index=True)

        # Stats équipes : toutes les ligues confondues
        all_leagues = TARGET_LEAGUES + TOP5_LEAGUES
        teams = fetch_team_stats(all_leagues)

    players_path = DATA_DIR / "players_raw.csv"
    teams_path   = DATA_DIR / "teams_raw.csv"

    players.to_csv(players_path, index=False)
    teams.to_csv(teams_path,     index=False)

    n_sec  = (players["tier"] == "secondary").sum() if "tier" in players.columns else len(players)
    n_top5 = (players["tier"] == "top5").sum()      if "tier" in players.columns else 0

    print(f"\n{len(players):,} joueurs total  -> {players_path}")
    print(f"  {n_sec:,} ligues secondaires  |  {n_top5:,} top 5 (remplacants)")
    print(f"{len(teams):,} equipes  -> {teams_path}")
    print("\nColonnes joueurs :", list(players.columns[:10]))
    print("Colonnes equipes :", list(teams.columns[:10]))


if __name__ == "__main__":
    main()
