"""
fetch_tm_missing.py  -  Recherche individuelle TM pour les joueurs sans donnees.

Cherche chaque joueur manquant via la recherche TM, valide par club,
extrait valeur marchande + fin de contrat, enrichit player_enrichment.csv.

    python scripts/fetch_tm_missing.py
"""

from __future__ import annotations

import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DATA_DIR  # noqa: E402

BASE_URL = "https://www.transfermarkt.de"
HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    "Referer": "https://www.transfermarkt.de/",
}
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).strip())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.lower()


def _get(url: str) -> requests.Response | None:
    for attempt in range(3):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(2 + attempt)
    return None


def _parse_mv(text: str) -> float | None:
    text = text.replace("\xa0", " ").strip()
    m = re.search(r"([\d,.]+)\s*Mio", text)
    if m:
        return float(m.group(1).replace(".", "").replace(",", ".")) * 1_000_000
    m = re.search(r"([\d,.]+)\s*Tsd", text)
    if m:
        return float(m.group(1).replace(".", "").replace(",", ".")) * 1_000
    return None


def _parse_date(text: str) -> str | None:
    m = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", text.strip())
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


def search_player(name: str, team: str, age: int | None = None) -> dict | None:
    """Recherche un joueur sur TM, valide par nom+age, retourne {mv, contract_end} ou None."""
    query = name.replace(" ", "+")
    url   = f"{BASE_URL}/schnellsuche/ergebnis/schnellsuche?query={query}&Kategorie=Spieler"
    r     = _get(url)
    if r is None:
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.select("table.items tbody tr.odd, table.items tbody tr.even")
    if not rows:
        return None

    team_norm = normalize(team)

    for row in rows[:5]:
        tds = row.find_all("td")
        if len(tds) < 7:
            continue

        row_name  = tds[2].get_text(strip=True) if len(tds) > 2 else ""
        row_club  = tds[3].get_text(strip=True) if len(tds) > 3 else ""
        row_age_s = tds[6].get_text(strip=True) if len(tds) > 6 else ""
        row_mv_td = next((td for td in reversed(tds) if "Mio" in td.get_text() or "Tsd" in td.get_text()), None)

        name_score = fuzz.token_sort_ratio(normalize(name), normalize(row_name))
        club_score = fuzz.token_sort_ratio(team_norm, normalize(row_club))

        # Validation age
        age_ok = False
        if age and row_age_s.isdigit():
            age_ok = abs(int(row_age_s) - age) <= 2

        # Accepter si : bon nom + (bon club OU age correspond)
        if name_score < 70:
            continue
        if club_score < 40 and not age_ok:
            continue
        # Si score nom trop bas et age absent, rejeter
        if name_score < 75 and not age_ok:
            continue

        mv = _parse_mv(row_mv_td.get_text(strip=True)) if row_mv_td else None

        # Fin de contrat + pied fort depuis le profil
        profile_link = row.select_one("a[href*='/profil/spieler/']")
        contract_end = None
        foot         = None
        if profile_link:
            profile_r = _get(BASE_URL + profile_link["href"])
            if profile_r:
                psoup = BeautifulSoup(profile_r.text, "html.parser")
                dates = [
                    span.get_text(strip=True)
                    for span in psoup.select("span.data-header__content")
                    if re.match(r"\d{2}\.\d{2}\.\d{4}", span.get_text(strip=True))
                    and "(" not in span.get_text()
                ]
                future = [d for d in dates if int(d.split(".")[-1]) >= 2025]
                if future:
                    latest = max(future, key=lambda d: (d.split(".")[-1], d.split(".")[1]))
                    contract_end = _parse_date(latest)
                # Pied fort
                for span in psoup.select("span.data-header__content"):
                    t = span.get_text(strip=True).lower()
                    if t in ("rechts", "links", "beidfussig", "beidf\xfcssig"):
                        foot = {"rechts": "Droit", "links": "Gauche"}.get(t, "Ambidextre")
                        break
            time.sleep(0.8)

        return {"market_value_eur": mv, "contract_end": contract_end, "foot": foot}

    return None


def main() -> None:
    enrich_path  = DATA_DIR / "player_enrichment.csv"
    players_path = DATA_DIR / "players_raw.csv"

    if not enrich_path.exists():
        print("player_enrichment.csv manquant. Lance d abord build_enrichment.py")
        sys.exit(1)

    enrich  = pd.read_csv(enrich_path)
    players = pd.read_csv(players_path)[["player", "team", "league"]]

    # Recuperer l age depuis players_raw
    players_full = pd.read_csv(players_path)
    age_col = next((c for c in ["age", "age_"] if c in players_full.columns), None)
    if age_col:
        players_full["_age_int"] = pd.to_numeric(
            players_full[age_col].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
        )
        players = players.merge(players_full[["player","league","_age_int"]], on=["player","league"], how="left")

    # Fusionner pour identifier les manquants
    merged = players.merge(
        enrich[["player", "league", "market_value_eur", "contract_end"]],
        on=["player", "league"], how="left"
    )
    missing = merged[merged["market_value_eur"].isna()].copy()
    print(f"{len(missing)} joueurs sans valeur marchande a rechercher sur TM")

    found = 0
    for i, (_, row) in enumerate(missing.iterrows()):
        if i % 20 == 0:
            print(f"  {i}/{len(missing)} ({found} trouves)...")

        age    = int(row["_age_int"]) if "_age_int" in row and pd.notna(row.get("_age_int")) else None
        result = search_player(row["player"], row["team"], age=age)
        if result and (result["market_value_eur"] or result["contract_end"] or result.get("foot")):
            # Mettre a jour dans enrich
            mask = (enrich["player"] == row["player"]) & (enrich["league"] == row["league"])
            if mask.any():
                if result["market_value_eur"]:
                    enrich.loc[mask, "market_value_eur"] = result["market_value_eur"]
                    sal_mask = mask & enrich["annual_gross_eur"].isna()
                    if sal_mask.any():
                        enrich.loc[sal_mask, "annual_gross_eur"] = result["market_value_eur"] * 0.22
                        enrich.loc[sal_mask, "salary_estimated"] = True
                if result["contract_end"]:
                    enrich.loc[mask, "contract_end"] = result["contract_end"]
                if result.get("foot"):
                    enrich.loc[mask, "foot"] = result["foot"]
            else:
                # Joueur absent de enrichment, on l ajoute
                new_row = {
                    "player": row["player"], "team": row["team"], "league": row["league"],
                    "market_value_eur": result["market_value_eur"],
                    "contract_end":     result["contract_end"],
                    "annual_gross_eur": (result["market_value_eur"] * 0.22
                                        if result["market_value_eur"] else np.nan),
                    "salary_estimated": bool(result["market_value_eur"]),
                }
                enrich = pd.concat([enrich, pd.DataFrame([new_row])], ignore_index=True)
            found += 1

        time.sleep(1.0)

    enrich.to_csv(enrich_path, index=False)

    mv_final  = enrich["market_value_eur"].notna().sum()
    ct_final  = enrich["contract_end"].notna().sum()
    sal_final = enrich["annual_gross_eur"].notna().sum()
    n         = len(players)
    print(f"\nTermine : {found} joueurs supplementaires trouves")
    print(f"Valeur marchande : {mv_final}/{n} ({100*mv_final//n}%)")
    print(f"Fin de contrat   : {ct_final}/{n} ({100*ct_final//n}%)")
    print(f"Salaire total    : {sal_final}/{n} ({100*sal_final//n}%)")
    print(f"-> {enrich_path}")


if __name__ == "__main__":
    main()
