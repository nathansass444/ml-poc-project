"""
collect_data.py  -  Telecharge les stats joueurs et equipes depuis FBref.

    python scripts/collect_data.py

Stats de base (soccerdata) : standard, shooting, playing_time, misc
Stats etendues (scraping direct) : passing, defense, possession, gca
Genere data/players_raw.csv et data/teams_raw.csv.
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
    from soccerdata.fbref import FBref, FBREF_API, _parse_table
    from soccerdata._common import standardize_colnames
    from lxml import html as lxml_html
except ImportError as e:
    print(f"Dependance manquante : {e}. Lancez : pip install soccerdata lxml")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers colonnes
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        cols = []
        for top, bot in df.columns:
            if "unnamed" in str(top).lower():
                cols.append(str(bot))
            else:
                cols.append(f"{top}_{bot}")
        df.columns = cols
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Stats etendues (passing / defense / possession / gca)
# Ces types ne sont pas dans soccerdata — on scrape FBref directement
# ─────────────────────────────────────────────────────────────────────────────

EXTRA_STAT_TYPES = ["passing", "defense", "possession", "gca"]


def _fetch_extra_stat(fbref: FBref, stat_type: str) -> pd.DataFrame:
    seasons = fbref.read_seasons()
    frames: list[pd.DataFrame] = []
    for (lkey, skey), season in seasons.iterrows():
        big_five = lkey == "Big 5 European Leagues Combined"
        filepath = fbref.data_dir / f"players_{lkey}_{skey}_{stat_type}.html"
        url = (
            FBREF_API
            + "/".join(season.url.split("/")[:-1])
            + f"/{stat_type}"
            + ("/players/" if big_five else "/")
            + season.url.split("/")[-1]
        )
        try:
            reader = fbref.get(url, filepath)
            tree   = lxml_html.parse(reader)
            tables = tree.xpath(f"//table[@id='stats_{stat_type}']")
            if not tables:
                continue
            df = _flatten_columns(_parse_table(tables[0]).pipe(standardize_colnames))
            df["league"] = lkey
            df["season"] = skey
            frames.append(df)
        except Exception as exc:
            print(f"    SKIP {lkey} {stat_type} ({exc})")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Fetch joueurs
# ─────────────────────────────────────────────────────────────────────────────

def fetch_player_stats(
    leagues: list[str],
    min_minutes: int = MIN_MINUTES,
    club_filter: str | None = None,
    tier: str = "secondary",
) -> pd.DataFrame:
    print(f"Fetch joueurs [{tier}] — {leagues}")
    fbref = sd.FBref(leagues=leagues, seasons=SEASON)

    # Stats de base via soccerdata
    base_types = ["standard", "shooting", "playing_time", "misc"]
    frames: list[pd.DataFrame] = []

    for stat in base_types:
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

    # Stats etendues via scraping direct
    for stat in EXTRA_STAT_TYPES:
        print(f"  stat_type={stat} (etendu)...")
        try:
            extra = _fetch_extra_stat(fbref, stat)
            if extra.empty:
                continue
            # Normaliser les cles de jointure
            rename = {"Player": "player", "Squad": "team", "Nation": "nation",
                      "Pos": "pos", "Age": "age"}
            extra.rename(columns={k: v for k, v in rename.items() if k in extra.columns}, inplace=True)
            # Garder seulement les nouvelles colonnes + cles
            keep_id   = [c for c in id_cols if c in extra.columns]
            new_cols  = [c for c in extra.columns if c not in merged.columns and c not in
                         ["Rk", "Born", "90s", "Age", "Pos", "Nation", "Matches", "age", "nation", "pos"]]
            extra_sub = extra[keep_id + new_cols].drop_duplicates(subset=keep_id)
            merged    = merged.merge(extra_sub, on=keep_id, how="left", suffixes=("", "_dup"))
            merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")], inplace=True)
            print(f"    +{len(new_cols)} colonnes : {new_cols[:5]}...")
        except Exception as exc:
            print(f"  SKIP {stat} ({exc})")

    # Filtre minutes
    min_col = next(
        (c for c in merged.columns if c.lower() == "playing time_min"),
        next((c for c in merged.columns if c.lower().endswith("_min") and "playing" in c.lower()), None),
    )
    if min_col:
        before  = len(merged)
        merged  = merged[merged[min_col] >= min_minutes].copy()
        print(f"  Filtre {min_minutes} min : {before} -> {len(merged)} joueurs")
    else:
        print("  WARN: colonne minutes non trouvee")

    if club_filter:
        team_col = next((c for c in merged.columns if c.lower() == "team"), None)
        if team_col:
            merged = merged[merged[team_col] == club_filter].copy()
            print(f"  Filtre club : {club_filter} -> {len(merged)} joueurs")

    merged["tier"] = tier
    return merged.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Fetch equipes
# ─────────────────────────────────────────────────────────────────────────────

def fetch_team_stats(leagues: list[str]) -> pd.DataFrame:
    print(f"Fetch equipes — {leagues}")
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
        raise RuntimeError("Aucune stat equipe recuperee.")

    id_cols = ["team", "league", "season"]
    merged  = frames[0]
    for other in frames[1:]:
        keys = [c for c in id_cols if c in merged.columns and c in other.columns]
        merged = merged.merge(other, on=keys, how="left", suffixes=("", "_dup"))
        merged.drop(columns=[c for c in merged.columns if c.endswith("_dup")], inplace=True)

    return merged.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TEST_MODE:
        print("MODE TEST : 1 ligue, 1 club")
        players = fetch_player_stats([TEST_LEAGUE], club_filter=TEST_CLUB, tier="secondary")
        teams   = fetch_team_stats([TEST_LEAGUE])
    else:
        print("MODE COMPLET : ligues secondaires + Top 5")

        players_secondary = fetch_player_stats(
            TARGET_LEAGUES, min_minutes=MIN_MINUTES, tier="secondary"
        )
        players_top5 = fetch_player_stats(
            TOP5_LEAGUES, min_minutes=MIN_MINUTES_TOP5, tier="top5"
        )
        players = pd.concat([players_secondary, players_top5], ignore_index=True)
        teams   = fetch_team_stats(TARGET_LEAGUES + TOP5_LEAGUES)

    players_path = DATA_DIR / "players_raw.csv"
    teams_path   = DATA_DIR / "teams_raw.csv"

    players.to_csv(players_path, index=False)
    teams.to_csv(teams_path, index=False)

    n_sec  = (players["tier"] == "secondary").sum() if "tier" in players.columns else len(players)
    n_top5 = (players["tier"] == "top5").sum()      if "tier" in players.columns else 0

    print(f"\n{len(players):,} joueurs ({n_sec:,} secondaires + {n_top5:,} top5) -> {players_path}")
    print(f"{len(teams):,} equipes -> {teams_path}")
    print(f"{len(players.columns)} colonnes joueurs | {len(teams.columns)} colonnes equipes")


if __name__ == "__main__":
    main()
