"""
build_enrichment.py  -  Construit data/player_enrichment.csv

Merge TM (valeur marchande + fin de contrat) et Capology (salaire)
sur les joueurs FBref en 3 passes :
  1. Exact match
  2. Normalized match (sans accents, lowercase)
  3. Fuzzy match dans la meme ligue (seuil 88, rapidfuzz)

Pour les joueurs avec valeur TM mais sans salaire Capology :
  salaire estime = market_value * SALARY_RATIO (calibre sur donnees connues).

    python scripts/build_enrichment.py
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DATA_DIR  # noqa: E402

SALARY_RATIO = 0.20   # approximation : salaire brut annuel ≈ 20% de la valeur marchande


# ─────────────────────────────────────────────────────────────────────────────

def normalize(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower()


def fuzzy_merge(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_key: str,
    right_key: str,
    value_cols: list[str],
    group_col: str | None = None,
    threshold: int = 88,
) -> pd.DataFrame:
    """
    Fuzzy-merge right into left sur les lignes de left ou left_key n'a pas encore de match.
    group_col : si fourni, le fuzzy match est restreint au meme groupe (ex : meme ligue).
    Retourne left avec value_cols remplis la ou possible.
    """
    right_norm = right.copy()
    right_norm["_key"] = right_norm[right_key].apply(normalize)

    if group_col:
        right_by_group: dict[str, dict] = {}
        for grp, sub in right_norm.groupby(group_col):
            right_by_group[str(grp)] = dict(zip(sub["_key"], sub[value_cols].values.tolist()))
    else:
        pool = dict(zip(right_norm["_key"], right_norm[value_cols].values.tolist()))

    results: list[dict] = []
    for _, row in left.iterrows():
        if group_col:
            grp_key = str(row.get(group_col, ""))
            pool = right_by_group.get(grp_key, {})
        if not pool:
            results.append({c: np.nan for c in value_cols})
            continue
        match = process.extractOne(
            normalize(str(row[left_key])),
            list(pool.keys()),
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
        )
        if match:
            vals = pool[match[0]]
            results.append(dict(zip(value_cols, vals)))
        else:
            results.append({c: np.nan for c in value_cols})

    return pd.DataFrame(results, index=left.index)


# ─────────────────────────────────────────────────────────────────────────────

def merge_with_fallback(
    players: pd.DataFrame,
    source: pd.DataFrame,
    source_name_col: str,
    value_cols: list[str],
    league_col_source: str | None = None,
) -> pd.DataFrame:
    """3-pass merge: exact -> normalized -> fuzzy."""
    players = players.copy()
    for c in value_cols:
        players[c] = np.nan

    source = source.copy()
    source["_norm"] = source[source_name_col].apply(normalize)
    players["_norm"] = players["player"].apply(normalize)

    # Pass 1 : exact match sur nom original
    src_exact = source.drop_duplicates(subset=[source_name_col])[[source_name_col] + value_cols]
    tmp = players[["player"]].merge(src_exact, left_on="player", right_on=source_name_col, how="left")
    for c in value_cols:
        col = c if c in tmp.columns else f"{c}_y"
        players[c] = tmp[col].values

    # Pass 2 : normalized match sur les non-matches
    missing = players[value_cols[0]].isna()
    if missing.any():
        src_norm = source.drop_duplicates(subset=["_norm"])[["_norm"] + value_cols]
        tmp2 = players.loc[missing, ["_norm"]].merge(src_norm, on="_norm", how="left")
        for c in value_cols:
            col = c if c in tmp2.columns else f"{c}_y"
            players.loc[missing, c] = tmp2[col].values

    # Pass 3 : fuzzy match sur les encore non-matches
    missing2 = players[value_cols[0]].isna()
    if missing2.any():
        group_col = "league" if league_col_source else None
        src_for_fuzzy = source.rename(columns={league_col_source: "league"}) if league_col_source else source
        fuzzy_result = fuzzy_merge(
            players[missing2][["player", "league"] if league_col_source else ["player"]],
            src_for_fuzzy,
            left_key="player",
            right_key=source_name_col,
            value_cols=value_cols,
            group_col=group_col,
        )
        for c in value_cols:
            players.loc[missing2, c] = fuzzy_result[c].values

    players.drop(columns=["_norm"], inplace=True, errors="ignore")
    return players


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    players_path = DATA_DIR / "players_raw.csv"
    tm_path      = DATA_DIR / "tm_players.csv"
    sal_path     = DATA_DIR / "salaries.csv"
    out_path     = DATA_DIR / "player_enrichment.csv"

    if not players_path.exists():
        print("players_raw.csv manquant.")
        sys.exit(1)

    players = pd.read_csv(players_path)[["player", "team", "league"]].copy()
    n = len(players)
    print(f"{n:,} joueurs a enrichir")

    # ── TM : valeur marchande + fin de contrat ─────────────────────────────
    if tm_path.exists():
        tm = pd.read_csv(tm_path)
        print("\nMerge TM (market_value_eur + contract_end)...")
        players = merge_with_fallback(
            players, tm,
            source_name_col="player_tm",
            value_cols=["market_value_eur", "contract_end"],
            league_col_source="league_fbref",
        )
        mv_ok = players["market_value_eur"].notna().sum()
        ct_ok = players["contract_end"].notna().sum()
        print(f"  Valeur marchande : {mv_ok:,}/{n} ({100*mv_ok//n}%)")
        print(f"  Fin de contrat   : {ct_ok:,}/{n} ({100*ct_ok//n}%)")
    else:
        print("tm_players.csv manquant — TM skip")
        players["market_value_eur"] = np.nan
        players["contract_end"]     = None

    # ── Capology : salaire annuel ──────────────────────────────────────────
    if sal_path.exists():
        sal = pd.read_csv(sal_path)
        print("\nMerge Capology (annual_gross_eur)...")
        players = merge_with_fallback(
            players, sal,
            source_name_col="player_cap",
            value_cols=["annual_gross_eur"],
            league_col_source="league_fbref",
        )
        sal_ok = players["annual_gross_eur"].notna().sum()
        print(f"  Salaire connu    : {sal_ok:,}/{n} ({100*sal_ok//n}%)")
    else:
        print("salaries.csv manquant — Capology skip")
        players["annual_gross_eur"] = np.nan

    # ── Estimation salaire depuis MV pour les joueurs sans salaire ─────────
    missing_sal  = players["annual_gross_eur"].isna()
    has_mv       = players["market_value_eur"].notna()
    to_estimate  = missing_sal & has_mv

    if to_estimate.any():
        # Calibrer le ratio sur les joueurs avec les deux valeurs
        both = players["annual_gross_eur"].notna() & players["market_value_eur"].notna()
        if both.sum() >= 10:
            ratio = (players.loc[both, "annual_gross_eur"] /
                     players.loc[both, "market_value_eur"]).median()
            ratio = float(np.clip(ratio, 0.05, 0.50))
        else:
            ratio = SALARY_RATIO
        print(f"\nEstimation salaire (ratio calibre = {ratio:.2f}) pour {to_estimate.sum():,} joueurs...")
        players.loc[to_estimate, "annual_gross_eur"] = (
            players.loc[to_estimate, "market_value_eur"] * ratio
        ).round(0)
        players.loc[to_estimate, "salary_estimated"] = True

    players["salary_estimated"] = players.get("salary_estimated", False).fillna(False)

    # ── Stats finales ──────────────────────────────────────────────────────
    mv_final  = players["market_value_eur"].notna().sum()
    sal_final = players["annual_gross_eur"].notna().sum()
    ct_final  = players["contract_end"].notna().sum()
    est_final = players["salary_estimated"].sum()
    print("\n" + "-"*50)
    print(f"Valeur marchande : {mv_final:,}/{n} ({100*mv_final//n}%)")
    print(f"Fin de contrat   : {ct_final:,}/{n} ({100*ct_final//n}%)")
    print(f"Salaire (reel)   : {sal_final - est_final:,}/{n}")
    print(f"Salaire (estime) : {est_final:,}/{n}")
    print(f"Salaire total    : {sal_final:,}/{n} ({100*sal_final//n}%)")

    players.to_csv(out_path, index=False)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
