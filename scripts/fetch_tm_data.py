"""
fetch_tm_data.py  -  Scrape valeur marchande + fin de contrat depuis Transfermarkt.

Genere data/tm_players.csv avec : player_tm, club_tm, market_value_eur, contract_end.
Necessite ~5-10 min selon le nombre de ligues et equipes.

    python scripts/fetch_tm_data.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DATA_DIR  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Mapping FBref league → TM wettbewerb id
# ─────────────────────────────────────────────────────────────────────────────

LEAGUE_TM_IDS = {
    "ENG-Championship":      "GB2",
    "FRA-Ligue 2":           "FR2",
    "GER-2. Bundesliga":     "L2",
    "ITA-Serie B":           "IT2",
    "NED-Eredivisie":        "NL1",
    "BEL-First Division A":  "BE1",
    "POR-Primeira Liga":     "PO1",
    "ESP-Segunda Division":  "ES2",
    "ENG-Premier League":    "GB1",
    "ESP-La Liga":           "ES1",
    "GER-Bundesliga":        "L1",
    "FRA-Ligue 1":           "FR1",
    "ITA-Serie A":           "IT1",
}

BASE_URL = "https://www.transfermarkt.de"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
    "Referer": "https://www.transfermarkt.de/",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=20)
            if r.status_code == 200:
                return r
            print(f"    HTTP {r.status_code} pour {url}")
        except Exception as e:
            print(f"    Erreur reseau ({e}) — tentative {attempt+1}/{retries}")
        time.sleep(2 + attempt * 2)
    return None


def _parse_market_value(text: str) -> float | None:
    """Convertit '1,20 Mio. €' ou '500 Tsd. €' en float EUR."""
    text = text.replace("\xa0", " ").strip()
    m = re.search(r"([\d,.]+)\s*Mio", text)
    if m:
        return float(m.group(1).replace(".", "").replace(",", ".")) * 1_000_000
    m = re.search(r"([\d,.]+)\s*Tsd", text)
    if m:
        return float(m.group(1).replace(".", "").replace(",", ".")) * 1_000
    return None


def _parse_contract_end(text: str) -> str | None:
    """Convertit '30.06.2028' (DD.MM.YYYY) en 'YYYY-MM-DD'."""
    text = text.strip()
    m = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", text)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Scraping par equipe
# ─────────────────────────────────────────────────────────────────────────────

def _scrape_team_squad(squad_path: str) -> list[dict]:
    """Scrape un kader TM (/club/kader/verein/ID/saison_id/2025/plus/1)."""
    url = BASE_URL + squad_path.rstrip("/") + "/plus/1"
    # Forcer saison 2025 si pas deja dans l URL
    if "saison_id" not in url:
        url = url + "/saison_id/2025/plus/1"

    r = _get(url)
    if r is None:
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    rows = soup.select("table.items tbody tr.odd, table.items tbody tr.even")
    players = []
    for row in rows:
        tds = row.find_all("td")
        if len(tds) < 12:
            continue
        # Nom : td avec class hauptlink (td[3])
        name_td = next((td for td in tds if "hauptlink" in (td.get("class") or [])), None)
        if name_td is None:
            continue
        name = name_td.get_text(strip=True)
        if not name:
            continue

        # Valeur marchande : dernier td avec class rechts hauptlink
        mv_td = tds[-1]
        mv = _parse_market_value(mv_td.get_text(strip=True))

        # Fin de contrat : avant-dernier td non vide avec format date DD.MM.YYYY
        contract_end = None
        for td in reversed(tds[:-1]):
            t = td.get_text(strip=True)
            if re.match(r"\d{2}\.\d{2}\.\d{4}", t):
                contract_end = _parse_contract_end(t)
                break

        # Pied fort : td contenant exactement rechts / links / beidfussig
        foot = None
        for td in tds:
            t = td.get_text(strip=True).lower()
            if t in ("rechts", "links", "beidfussig", "beidf\xfcssig", "beidf\xfc\xdfig"):
                foot = {"rechts": "Droit", "links": "Gauche"}.get(t, "Ambidextre")
                break

        players.append({
            "player_tm":        name,
            "market_value_eur": mv,
            "contract_end":     contract_end,
            "foot":             foot,
        })
    return players


# ─────────────────────────────────────────────────────────────────────────────
# Scraping par ligue
# ─────────────────────────────────────────────────────────────────────────────

def _get_team_squad_paths(wettbewerb_id: str) -> list[str]:
    """Retourne les paths /kader/verein/... de toutes les equipes d une ligue."""
    url = f"{BASE_URL}/a/startseite/wettbewerb/{wettbewerb_id}"
    r = _get(url)
    if r is None:
        return []
    soup = BeautifulSoup(r.text, "html.parser")
    paths = []
    seen = set()
    for a in soup.select("a[href*='/kader/verein/']"):
        href = a["href"]
        if href not in seen and "saison_id" in href:
            seen.add(href)
            paths.append(href)
    return paths


def fetch_league(league_name: str, wettbewerb_id: str) -> pd.DataFrame:
    print(f"\n  Ligue : {league_name} [{wettbewerb_id}]")
    squad_paths = _get_team_squad_paths(wettbewerb_id)
    print(f"    {len(squad_paths)} equipes trouvees")

    all_players: list[dict] = []
    for path in squad_paths:
        team_name = path.split("/")[1]
        players = _scrape_team_squad(path)
        for p in players:
            p["club_tm"] = team_name
        all_players.extend(players)
        print(f"    {team_name}: {len(players)} joueurs")
        time.sleep(1.2)

    df = pd.DataFrame(all_players)
    df["league_fbref"] = league_name
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "tm_players.csv"

    frames: list[pd.DataFrame] = []
    for league_name, wettbewerb_id in LEAGUE_TM_IDS.items():
        try:
            df = fetch_league(league_name, wettbewerb_id)
            if not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  ERREUR {league_name}: {e}")
        time.sleep(2)

    if not frames:
        print("Aucune donnee recuperee.")
        return

    result = pd.concat(frames, ignore_index=True)
    result.drop_duplicates(subset=["player_tm", "club_tm"], inplace=True)
    result.to_csv(out_path, index=False)

    n_mv  = result["market_value_eur"].notna().sum()
    n_ctr = result["contract_end"].notna().sum()
    print(f"\n{len(result):,} joueurs -> {out_path}")
    print(f"  Valeurs marchandes : {n_mv:,} | Fins de contrat : {n_ctr:,}")


if __name__ == "__main__":
    main()
