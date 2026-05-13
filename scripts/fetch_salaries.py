"""
fetch_salaries.py  -  Scrape salaires depuis Capology.com

Genere data/salaries.csv avec : player_cap, annual_gross_eur, league_fbref.
    python scripts/fetch_salaries.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DATA_DIR  # noqa: E402

LEAGUE_URLS = {
    "ENG-Premier League":   "https://www.capology.com/uk/premier-league/salaries/",
    "ESP-La Liga":          "https://www.capology.com/es/la-liga/salaries/",
    "GER-Bundesliga":       "https://www.capology.com/de/1-bundesliga/salaries/",
    "FRA-Ligue 1":          "https://www.capology.com/fr/ligue-1/salaries/",
    "ITA-Serie A":          "https://www.capology.com/it/serie-a/salaries/",
    "ENG-Championship":     "https://www.capology.com/uk/championship/salaries/",
    "FRA-Ligue 2":          "https://www.capology.com/fr/ligue-2/salaries/",
    "ITA-Serie B":          "https://www.capology.com/it/serie-b/salaries/",
    "POR-Primeira Liga":    "https://www.capology.com/pt/primeira-liga/salaries/",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _parse_league(html: str, league_name: str) -> list[dict]:
    m = re.search(r"var data = (\[.*?\]);", html, re.DOTALL)
    if not m:
        return []

    raw = m.group(1)
    blocks = re.split(r"\},\s*\{", raw)
    players = []

    for block in blocks:
        # Player name — strip HTML tags from 'name' field
        name_m = re.search(r"'name':\s*\"(.*?)\"", block, re.DOTALL)
        if not name_m:
            continue
        name = re.sub(r"<[^>]+>", "", name_m.group(1)).strip()
        if not name:
            continue

        # Annual gross EUR — first number in accounting.formatMoney for annual_gross_eur
        sal_m = re.search(r"'annual_gross_eur':\s*accounting\.formatMoney\(\"(\d+)\"", block)
        salary = int(sal_m.group(1)) if sal_m else None

        players.append({
            "player_cap":        name,
            "annual_gross_eur":  salary,
            "league_fbref":      league_name,
        })

    return players


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "salaries.csv"

    sess = requests.Session()
    sess.headers.update(HEADERS)

    frames: list[pd.DataFrame] = []

    for league_name, url in LEAGUE_URLS.items():
        print(f"  {league_name}...", end=" ", flush=True)
        try:
            r = sess.get(url, timeout=20)
            if r.status_code != 200:
                print(f"HTTP {r.status_code} — skip")
                continue
            players = _parse_league(r.text, league_name)
            print(f"{len(players)} joueurs")
            if players:
                frames.append(pd.DataFrame(players))
        except Exception as e:
            print(f"ERREUR ({e})")
        time.sleep(1.5)

    if not frames:
        print("Aucune donnee recuperee.")
        return

    result = pd.concat(frames, ignore_index=True)
    result.drop_duplicates(subset=["player_cap", "league_fbref"], inplace=True)
    result.to_csv(out_path, index=False)

    n_sal = result["annual_gross_eur"].notna().sum()
    print(f"\n{len(result):,} joueurs -> {out_path}")
    print(f"  Salaires disponibles : {n_sal:,}")


if __name__ == "__main__":
    main()
