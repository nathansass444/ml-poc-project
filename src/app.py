"""
app.py  -  Scout Intelligence Tool

Workflow :
  1. Selectionner l equipe qui recrute
  2. Selectionner le joueur a remplacer (ou le poste a pourvoir)
  3. L app retourne 5-10 remplacants filtrables par tier / ligue
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import DATA_DIR, MODELS_DIR   # noqa: E402
from features import (                    # noqa: E402
    prepare_player_features,
    prepare_team_features,
    get_role_label,
    label_team_cluster,
)


# ─────────────────────────────────────────────────────────────────────────────
# Chargement donnees et modeles (cache Streamlit)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    p = DATA_DIR / "players_raw.csv"
    t = DATA_DIR / "teams_raw.csv"
    if not p.exists() or not t.exists():
        return None, None
    players = pd.read_csv(p)
    pos_col = next((c for c in ["pos", "pos_", "position"] if c in players.columns), "pos_")
    players["role"] = players[pos_col].apply(get_role_label)
    # Normaliser les colonnes avec underscore final
    for col_under, col_clean in [("age_", "age"), ("nation_", "nation"), ("pos_", "pos")]:
        if col_under in players.columns and col_clean not in players.columns:
            players[col_clean] = players[col_under]

    # Pre-calcul des stats per-90 pour affichage dans les cartes
    d = players["Playing Time_90s"].clip(lower=1)
    for feat, raw in {
        "int_per90":  "Performance_Int",
        "tklw_per90": "Performance_TklW",
        "fls_per90":  "Performance_Fls",
        "fld_per90":  "Performance_Fld",
        "crs_per90":  "Performance_Crs",
        "off_per90":  "Performance_Off",
        "crd_per90":  "Performance_CrdY",
    }.items():
        if raw in players.columns:
            players[feat] = pd.to_numeric(players[raw], errors="coerce") / d

    # Merge donnees Transfermarkt (valeur marchande + fin de contrat)
    tm_path = DATA_DIR / "tm_players.csv"
    if tm_path.exists():
        tm = pd.read_csv(tm_path)[["player_tm", "market_value_eur", "contract_end"]]
        tm = tm.drop_duplicates(subset=["player_tm"])
        players = players.merge(tm, left_on="player", right_on="player_tm", how="left")
        players.drop(columns=["player_tm"], inplace=True, errors="ignore")
    else:
        players["market_value_eur"] = None
        players["contract_end"]     = None

    # Merge salaires Capology
    sal_path = DATA_DIR / "salaries.csv"
    if sal_path.exists():
        sal = pd.read_csv(sal_path)[["player_cap", "annual_gross_eur", "league_fbref"]]
        sal = sal.drop_duplicates(subset=["player_cap", "league_fbref"])
        players = players.merge(
            sal, left_on=["player", "league"], right_on=["player_cap", "league_fbref"], how="left"
        )
        players.drop(columns=["player_cap", "league_fbref"], inplace=True, errors="ignore")
    else:
        players["annual_gross_eur"] = None

    return players, pd.read_csv(t)


@st.cache_resource
def load_models():
    try:
        kmeans   = joblib.load(MODELS_DIR / "kmeans_team.joblib")
        rf       = joblib.load(MODELS_DIR / "rf_role.joblib")
        knn_data = joblib.load(MODELS_DIR / "knn_similarity.joblib")
        return kmeans, rf, knn_data
    except FileNotFoundError:
        return None, None, None


def _fmt_salary(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    val = float(val)
    if val >= 1_000_000:
        return f"{val/1_000_000:.1f}M€/an"
    if val >= 1_000:
        return f"{val/1_000:.0f}K€/an"
    return f"{int(val)}€/an"


def _fmt_market_value(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    val = float(val)
    if val >= 1_000_000:
        return f"{val/1_000_000:.1f}M€"
    if val >= 1_000:
        return f"{val/1_000:.0f}K€"
    return f"{int(val)}€"


def _fmt_contract_end(date_str) -> str:
    if not date_str or (isinstance(date_str, float) and np.isnan(date_str)):
        return "N/A"
    try:
        from datetime import date
        d = date.fromisoformat(str(date_str))
        months = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]
        return f"{months[d.month-1]} {d.year}"
    except Exception:
        return str(date_str)


# ─────────────────────────────────────────────────────────────────────────────
# Radar chart
# ─────────────────────────────────────────────────────────────────────────────

RADAR_FEATURES = [
    "Per 90 Minutes_Gls",
    "Per 90 Minutes_Ast",
    "Standard_Sh/90",
    "Standard_SoT/90",
    "int_per90",
    "tklw_per90",
    "fld_per90",
    "crs_per90",
]

RADAR_LABELS = {
    "Per 90 Minutes_Gls": "Buts/90",
    "Per 90 Minutes_Ast": "Passes D/90",
    "Standard_Sh/90":     "Tirs/90",
    "Standard_SoT/90":    "Tirs cadres/90",
    "int_per90":          "Interceptions/90",
    "tklw_per90":         "Tacles/90",
    "fld_per90":          "Fautes subies/90",
    "crs_per90":          "Centres/90",
}


def radar_chart(profiles: dict[str, pd.Series]) -> go.Figure:
    features = [f for f in RADAR_FEATURES if any(f in p.index for p in profiles.values())]
    labels   = [RADAR_LABELS.get(f, f) for f in features]
    colors   = ["#00d4ff", "#ff6b35", "#7bc67e", "#f7b731", "#a29bfe"]

    # Normalisation min-max sur les features communes
    all_vals = pd.DataFrame({n: [float(p.get(f, 0)) for f in features]
                              for n, p in profiles.items()}, index=features).T
    mins = all_vals.min()
    maxs = all_vals.max()
    rng  = (maxs - mins).replace(0, 1)
    norm = (all_vals - mins) / rng

    fig = go.Figure()
    for i, (name, _) in enumerate(profiles.items()):
        vals = norm.loc[name].tolist()
        vals.append(vals[0])
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=labels + [labels[0]],
            fill="toself",
            name=name,
            line_color=colors[i % len(colors)],
            opacity=0.75,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=420,
        margin=dict(t=30, b=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Recherche de remplacants via KNN
# ─────────────────────────────────────────────────────────────────────────────

def find_replacements(
    query_player_idx: int,
    players_df: pd.DataFrame,
    knn_data: dict,
    role_filter: str,
    exclude_leagues: list[str],
    tier_filter: str,
    n: int = 10,
) -> pd.DataFrame:
    scaler       = knn_data["scaler"]
    X_scaled     = knn_data["X_scaled"]
    player_index = knn_data["player_index"]

    # Vecteur du joueur a remplacer
    query_vec = X_scaled[query_player_idx].reshape(1, -1)

    # Similarite cosinus avec tous les joueurs
    sims = cosine_similarity(query_vec, X_scaled)[0]

    results = player_index.copy()
    results["similarity"] = sims
    results = results.merge(
        players_df[["player", "team", "league", "role", "tier",
                    "Playing Time_Min", "age", "nation"]].drop_duplicates("player"),
        on=["player", "team", "league"],
        how="left",
    )

    # Filtres
    results = results[results.index != query_player_idx]          # exclure le joueur lui-meme
    results = results[results["role"] == role_filter]             # meme role
    if exclude_leagues:
        results = results[~results["league"].isin(exclude_leagues)]
    if tier_filter != "Tous":
        results = results[results["tier"] == ("secondary" if tier_filter == "Ligues secondaires" else "top5")]

    return results.nlargest(n, "similarity").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# App principale
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> None:
    st.set_page_config(
        page_title="Scout Intelligence Tool",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
        .main-title  { font-size:2.2rem; font-weight:800; color:#00d4ff; }
        .subtitle    { color:#ccc; margin-top:-8px; margin-bottom:20px; }
        .player-card {
            background:#1e1e2e; border-radius:10px;
            padding:16px 18px 10px 18px; margin:8px 0;
            border-left:4px solid #00d4ff;
        }
        .player-card.top5 { border-left-color:#ff6b35; }
        .player-name {
            font-size:1.15rem; font-weight:700;
            color:#ffffff; letter-spacing:0.3px;
        }
        .player-meta { font-size:0.88rem; color:#cccccc; margin-top:3px; }
        .sim-bar-wrap {
            background:#2e2e3e; border-radius:6px;
            height:8px; margin:8px 0 4px 0; overflow:hidden;
        }
        .sim-bar { height:8px; border-radius:6px; }
        .sim-label { font-size:0.82rem; font-weight:600; }
        .tier-badge-top5 {
            background:#ff6b3522; color:#ff6b35;
            border:1px solid #ff6b35; border-radius:4px;
            font-size:0.72rem; padding:1px 6px; margin-left:8px;
            vertical-align:middle;
        }
        .metric-card {
            background:#1e1e2e; border-radius:10px;
            padding:14px; text-align:center;
        }
        .metric-value { font-size:1.5rem; font-weight:700; color:#00d4ff; }
        .metric-label { font-size:0.82rem; color:#aaa; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-title">⚽ Scout Intelligence Tool</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Trouve le remplacant ideal dans les ligues secondaires europeennes — saison 2025-26</p>',
        unsafe_allow_html=True,
    )

    players_df, teams_df = load_data()
    kmeans, rf, knn_data = load_models()

    if players_df is None:
        st.error("Donnees manquantes. Lance : python scripts/collect_data.py")
        st.stop()
    if kmeans is None:
        st.error("Modeles manquants. Lance : python scripts/train_models.py")
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Parametres")

        # 1. Equipe qui recrute
        team_list = sorted(teams_df["team"].dropna().unique().tolist())
        selected_team = st.selectbox("Equipe qui recrute", team_list)

        team_league = teams_df.loc[teams_df["team"] == selected_team, "league"]
        team_league_val = team_league.iloc[0] if len(team_league) > 0 else ""

        # 2. Joueur a remplacer (filtre par equipe)
        team_players = players_df[players_df["team"] == selected_team]["player"].dropna().sort_values().tolist()
        if not team_players:
            # fallback : chercher dans tout le dataset par nom d equipe approximatif
            team_players = players_df["player"].dropna().sort_values().tolist()

        selected_player = st.selectbox("Joueur a remplacer", team_players)

        # 3. Filtres
        st.markdown("---")
        st.markdown("**Filtres**")

        tier_filter = st.radio(
            "Source des remplacants",
            ["Tous", "Ligues secondaires", "Top 5 (remplacants)"],
            index=0,
        )

        exclude_own_league = st.checkbox("Exclure la ligue de l equipe", value=False)
        n_recommendations  = st.slider("Nombre de candidats", 5, 20, 10)

        st.markdown("---")
        st.caption("FBref via soccerdata | 2025-26 | 4 759 joueurs")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["Remplacants", "Style d equipe", "A propos"])

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 1 — Remplacants
    # ═════════════════════════════════════════════════════════════════════════
    with tab1:
        # Style de l equipe
        team_row = teams_df[teams_df["team"] == selected_team]
        if team_row.empty:
            st.warning("Equipe introuvable.")
            st.stop()

        X_team     = prepare_team_features(team_row)
        cluster_id = int(kmeans.predict(X_team.values)[0])
        style_name = label_team_cluster(cluster_id)

        # Infos joueur cible
        player_row = players_df[players_df["player"] == selected_player]
        if player_row.empty:
            st.warning(f"Joueur {selected_player} introuvable.")
            st.stop()

        player_role = player_row["role"].iloc[0]
        player_idx  = player_row.index[0]

        # Metriques rapides
        c1, c2, c3, c4 = st.columns(4)
        mins_played = int(player_row["Playing Time_Min"].iloc[0]) if "Playing Time_Min" in player_row else "?"
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{selected_player}</div>'
                    f'<div class="metric-label">Joueur a remplacer</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{style_name}</div>'
                    f'<div class="metric-label">Style {selected_team}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value">{player_role.capitalize()}</div>'
                    f'<div class="metric-label">Role detecte</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-card"><div class="metric-value">{mins_played}</div>'
                    f'<div class="metric-label">Minutes 2025-26</div></div>', unsafe_allow_html=True)

        st.markdown("---")

        # Recherche
        exclude_leagues = [team_league_val] if exclude_own_league and team_league_val else []

        replacements = find_replacements(
            query_player_idx=player_idx,
            players_df=players_df,
            knn_data=knn_data,
            role_filter=player_role,
            exclude_leagues=exclude_leagues,
            tier_filter=tier_filter,
            n=n_recommendations,
        )

        if replacements.empty:
            st.warning("Aucun remplacant trouve avec ces filtres.")
            st.stop()

        # Stats a afficher selon le role
        ROLE_STATS: dict[str, list[tuple[str, str]]] = {
            "forward":    [
                ("Buts/90",      "Per 90 Minutes_Gls"),
                ("Passes D/90",  "Per 90 Minutes_Ast"),
                ("Tirs/90",      "Standard_Sh/90"),
                ("Cadres/90",    "Standard_SoT/90"),
                ("Hors-jeu/90",  "off_per90"),
            ],
            "midfielder": [
                ("Buts/90",       "Per 90 Minutes_Gls"),
                ("Passes D/90",   "Per 90 Minutes_Ast"),
                ("Tacles/90",     "tklw_per90"),
                ("Interc./90",    "int_per90"),
                ("Centres/90",    "crs_per90"),
            ],
            "defender":   [
                ("Tacles/90",    "tklw_per90"),
                ("Interc./90",   "int_per90"),
                ("F. subies/90", "fld_per90"),
                ("Buts/90",      "Per 90 Minutes_Gls"),
                ("Passes D/90",  "Per 90 Minutes_Ast"),
            ],
            "goalkeeper": [
                ("Minutes",  "Playing Time_Min"),
                ("Matchs",   "Playing Time_MP"),
                ("Cartons",  "crd_per90"),
            ],
        }
        stats_to_show = ROLE_STATS.get(player_role, ROLE_STATS["midfielder"])

        # Stats du joueur de reference
        ref_row = players_df[players_df["player"] == selected_player].iloc[0]

        st.subheader(f"Top {len(replacements)} remplacants pour {selected_player}")
        st.caption("Bordure bleue = ligue secondaire  |  Bordure orange = Top 5 (peu utilise)")

        for i, (_, row) in enumerate(replacements.iterrows()):
            sim_pct    = int(row["similarity"] * 100)
            sim_color  = "#7bc67e" if sim_pct >= 80 else "#f7b731" if sim_pct >= 60 else "#e74c3c"
            tier_cls   = "top5" if row.get("tier") == "top5" else ""
            tier_badge = '<span class="tier-badge-top5">TOP 5</span>' if tier_cls == "top5" else ""
            age        = str(row.get("age", "?")).split("-")[0]
            nation     = row.get("nation", "")
            mins       = int(row["Playing Time_Min"]) if pd.notna(row.get("Playing Time_Min")) else "?"
            bar_width  = sim_pct

            # Donnees TM + Capology (depuis players_df pour avoir le merge complet)
            cand_full_row = players_df[players_df["player"] == row["player"]]
            cand_data = cand_full_row.iloc[0] if not cand_full_row.empty else pd.Series(dtype=object)
            mv_str  = _fmt_market_value(cand_data.get("market_value_eur"))
            ct_str  = _fmt_contract_end(cand_data.get("contract_end"))
            sal_str = _fmt_salary(cand_data.get("annual_gross_eur"))

            # En-tete de carte
            st.markdown(
                f'<div class="player-card {tier_cls}">'
                f'<span class="player-name">#{i+1} &nbsp;{row["player"]}</span>{tier_badge}<br>'
                f'<span class="player-meta">{row["team"]} &nbsp;·&nbsp; {row["league"]}</span><br>'
                f'<span class="player-meta">'
                f'{age} ans &nbsp;·&nbsp; {nation} &nbsp;·&nbsp; {mins} min'
                f' &nbsp;·&nbsp; <b style="color:#00d4ff">{mv_str}</b>'
                f' &nbsp;·&nbsp; <b style="color:#7bc67e">{sal_str}</b>'
                f' &nbsp;·&nbsp; Contrat : <b style="color:#f7b731">{ct_str}</b>'
                f'</span>'
                f'<div class="sim-bar-wrap"><div class="sim-bar" '
                f'style="width:{bar_width}%; background:{sim_color};"></div></div>'
                f'<span class="sim-label" style="color:{sim_color}">Similarite : {sim_pct}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Stats comparatives (candidat vs joueur remplace)
            cand_full = players_df[players_df["player"] == row["player"]]
            if not cand_full.empty:
                cand_row = cand_full.iloc[0]
                cols = st.columns(len(stats_to_show))
                for col, (label, col_name) in zip(cols, stats_to_show):
                    cand_val = float(cand_row.get(col_name, 0) or 0)
                    ref_val  = float(ref_row.get(col_name, 0) or 0)
                    delta    = round(cand_val - ref_val, 2)
                    col.metric(
                        label=label,
                        value=f"{cand_val:.2f}",
                        delta=f"{delta:+.2f} vs ref",
                        delta_color="normal",
                    )
            st.markdown("")   # espace entre les cartes

        # Radar chart top 3
        st.markdown("### Comparaison radar — Top 3")

        top3_indices = replacements.head(3).index.tolist()
        profiles = {selected_player: prepare_player_features(player_row).iloc[0]}

        for idx in top3_indices:
            name    = replacements.loc[idx, "player"]
            p_row   = players_df[players_df["player"] == name].head(1)
            if not p_row.empty:
                profiles[name] = prepare_player_features(p_row).iloc[0]

        if len(profiles) >= 2:
            st.plotly_chart(radar_chart(profiles), use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 2 — Style d equipe
    # ═════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("Analyse du style de jeu")

        all_teams      = sorted(teams_df["team"].dropna().unique())
        compare_teams  = st.multiselect(
            "Selectionne 2 a 4 equipes",
            all_teams,
            default=[selected_team] + [t for t in all_teams if t != selected_team][:2],
            max_selections=4,
        )

        if len(compare_teams) < 2:
            st.info("Selectionne au moins 2 equipes.")
        else:
            rows        = teams_df[teams_df["team"].isin(compare_teams)]
            X_compare   = prepare_team_features(rows)
            cluster_ids = kmeans.predict(X_compare.values)

            cols = st.columns(len(compare_teams))
            for i, (team, cid) in enumerate(zip(compare_teams, cluster_ids)):
                with cols[i]:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value" style="font-size:1.1rem">{team}</div>'
                        f'<div class="metric-label">{label_team_cluster(int(cid))}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        st.markdown("### Distribution des styles — toutes les equipes")
        all_X       = prepare_team_features(teams_df)
        all_labels  = kmeans.predict(all_X.values)
        counts      = pd.Series(all_labels).value_counts().sort_index()
        style_names = [label_team_cluster(i) for i in counts.index]

        fig = go.Figure(go.Bar(
            x=style_names, y=counts.values,
            marker_color=["#00d4ff", "#ff6b35", "#7bc67e", "#f7b731"],
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"), yaxis_title="Nb equipes", height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ═════════════════════════════════════════════════════════════════════════
    # TAB 3 — A propos
    # ═════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
        ### Scout Intelligence Tool — Saison 2025-26

        **Probleme resolu :** Les clubs a petit budget n ont pas les moyens de payer
        des plateformes comme Wyscout. Cet outil repond a la question :

        > *Etant donne un joueur qui quitte mon club, qui peut le remplacer
        > dans mon budget et avec mon style de jeu ?*

        ---

        ### Pipeline
        1. **KMeans** — classe chaque equipe par style (possession, pressing, direct, bloc bas)
        2. **Random Forest** — identifie le role tactique d un joueur depuis ses stats
        3. **KNN cosinus** — mesure la similarite entre le joueur a remplacer et les candidats

        ### Donnees
        - FBref via soccerdata | 4 759 joueurs | 250 equipes
        - 8 ligues secondaires + Top 5 (joueurs < 900 min = potentiellement libres)
        - Saison 2025-26 — pertinence directe pour le mercato ete 2026

        ### Limites
        > La data ne remplace pas l oeil du scout. Elle reduit le vivier
        > de 4 000 joueurs a 10 candidats. Le scout garde le dernier mot.
        """)


if __name__ == "__main__":
    build_app()
