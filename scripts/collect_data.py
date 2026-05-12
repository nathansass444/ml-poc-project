"""
collect_data.py  –  À exécuter UNE FOIS pour télécharger et préparer les données.

    python scripts/collect_data.py

Nécessite : pip install soccerdata pandas
Les fichiers générés dans data/ sont ensuite utilisés par src/data.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── allow running from any location ──────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import DATA_DIR, TARGET_LEAGUES, SEASON, MIN_MINUTES  # noqa: E402

try:
    import soccerdata as sd
except ImportError:
    print("❌  soccerdata non installé. Lancez : pip install soccerdata")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Player stats (standard + advanced)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_team_stats() -> pd.DataFrame:
    print("📥  Téléchargement des stats équipes via FBref …")
    fbref = sd.FBref(leagues="Big 5 European Leagues Combined", seasons=SEASON)

    df = fbref.read_team_season_stats(stat_type="standard")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(c).strip("_") for c in df.columns]

    return df.reset_index().reset_index(drop=True)

    std   = fbref.read_player_season_stats(stat_type="standard")
    pass_ = fbref.read_player_season_stats(stat_type="shooting")
    def_  = fbref.read_player_season_stats(stat_type="misc")
    poss  = fbref.read_player_season_stats(stat_type="playing_time")

    # Flatten MultiIndex columns if present
    for df in [std, pass_, def_, poss]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(c).strip("_") for c in df.columns]

    # Merge on player + team + league + season
    merge_keys = ["player", "team", "league", "season", "position", "nationality", "age"]
    merge_keys = [k for k in merge_keys if k in std.columns]

    df = std.copy()
    for extra in [pass_, def_, poss]:
        extra_keys = [k for k in merge_keys if k in extra.columns]
        df = df.merge(extra, on=extra_keys, how="left", suffixes=("", "_dup"))
        dup_cols = [c for c in df.columns if c.endswith("_dup")]
        df.drop(columns=dup_cols, inplace=True)

    # Filter minimum minutes
    minutes_col = next((c for c in df.columns if "minutes" in c.lower() or "min" in c.lower()), None)
    if minutes_col:
        df = df[df[minutes_col] >= MIN_MINUTES].copy()

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Team stats (collective style of play)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_team_stats() -> pd.DataFrame:
    print("📥  Téléchargement des stats équipes via FBref …")
    fbref = sd.FBref(leagues="Big 5 European Leagues Combined", seasons=SEASON)

    df = fbref.read_team_season_stats(stat_type="standard")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(c).strip("_") for c in df.columns]

    return df.reset_index().reset_index(drop=True)

    std   = fbref.read_team_season_stats(stat_type="standard")
    pass_ = fbref.read_team_season_stats(stat_type="shooting")
    def_  = fbref.read_team_season_stats(stat_type="misc")
    poss  = fbref.read_team_season_stats(stat_type="playing_time")
    misc  = fbref.read_team_season_stats(stat_type="keeper")

    for df in [std, pass_, def_, poss, misc]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(c).strip("_") for c in df.columns]

    merge_keys = ["team", "league", "season"]
    merge_keys = [k for k in merge_keys if k in std.columns]

    df = std.copy()
    for extra in [pass_, def_, poss, misc]:
        extra_keys = [k for k in merge_keys if k in extra.columns]
        df = df.merge(extra, on=extra_keys, how="left", suffixes=("", "_dup"))
        dup_cols = [c for c in df.columns if c.endswith("_dup")]
        df.drop(columns=dup_cols, inplace=True)

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Save
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    players = fetch_player_stats()
    teams   = fetch_team_stats()

    players_path = DATA_DIR / "players_raw.csv"
    teams_path   = DATA_DIR / "teams_raw.csv"

    players.to_csv(players_path, index=False)
    teams.to_csv(teams_path,     index=False)

    print(f"✅  {len(players):,} joueurs sauvegardés  →  {players_path}")
    print(f"✅  {len(teams):,}  équipes sauvegardées →  {teams_path}")
    print("\nColonnes joueurs :", list(players.columns[:15]), "…")
    print("Colonnes équipes :", list(teams.columns[:15]),   "…")


if __name__ == "__main__":
    main()
