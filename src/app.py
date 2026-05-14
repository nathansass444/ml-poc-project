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

    # Merge enrichissement (TM + Capology + estimation) pre-calcule
    enrich_path = DATA_DIR / "player_enrichment.csv"
    if enrich_path.exists():
        enrich = pd.read_csv(enrich_path)[
            ["player", "team", "league", "market_value_eur", "contract_end",
             "annual_gross_eur", "salary_estimated"]
        ].drop_duplicates(subset=["player", "team", "league"])
        players = players.merge(enrich, on=["player", "team", "league"], how="left")
    else:
        players["market_value_eur"]  = None
        players["contract_end"]      = None
        players["annual_gross_eur"]  = None
        players["salary_estimated"]  = False

    players["salary_estimated"] = players["salary_estimated"].fillna(False)

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
    budget_max_eur: float | None = None,
    include_unknown_mv: bool = True,
    age_min: int = 15,
    age_max: int = 40,
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
        players_df[["player", "team", "league", "role",
                    "Playing Time_Min", "age", "nation",
                    "market_value_eur"]].drop_duplicates("player"),
        on=["player", "team", "league"],
        how="left",
    )

    # Colonne age numerique pour filtrage
    results["_age_int"] = pd.to_numeric(
        results["age"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )

    # Filtres
    results = results[results.index != query_player_idx]          # exclure le joueur lui-meme
    results = results[results["role"] == role_filter]             # meme role
    if exclude_leagues:
        results = results[~results["league"].isin(exclude_leagues)]
    if tier_filter != "Tous":
        results = results[results["tier"] == ("secondary" if tier_filter == "Ligues secondaires" else "top5")]

    # Filtre age
    if age_min > 15 or age_max < 40:
        age_known = results["_age_int"].notna()
        in_range  = results["_age_int"].between(age_min, age_max)
        results   = results[~age_known | in_range]

    # Filtre budgetaire
    if budget_max_eur is not None:
        has_mv   = results["market_value_eur"].notna()
        in_budget = results["market_value_eur"] <= budget_max_eur
        if include_unknown_mv:
            results = results[~has_mv | in_budget]   # inclut N/A + sous budget
        else:
            results = results[has_mv & in_budget]    # uniquement ceux sous budget connu

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

        # Budget par defaut = valeur marchande du joueur a remplacer (ou 20M)
        ref_player_row = players_df[players_df["player"] == selected_player]
        ref_mv = None
        if not ref_player_row.empty:
            mv_raw = ref_player_row["market_value_eur"].iloc[0]
            if pd.notna(mv_raw):
                ref_mv = float(mv_raw)
        default_budget_m = round((ref_mv / 1_000_000) if ref_mv else 20.0, 1)

        # 3. Filtres
        st.markdown("---")
        st.markdown("**Filtres**")

        tier_filter = st.radio(
            "Source des remplacants",
            ["Tous", "Ligues secondaires", "Top 5 (remplacants)"],
            index=0,
        )

        exclude_own_league = st.checkbox("Exclure la ligue de l equipe", value=False)

        st.markdown("**Budget max (valeur marchande)**")
        budget_max_m = st.slider(
            "Budget max (M€)",
            min_value=0.1, max_value=200.0,
            value=float(min(default_budget_m, 200.0)),
            step=0.5,
            format="%.1f M€",
        )
        include_unknown_mv = st.checkbox("Inclure joueurs sans valeur connue", value=True)

        st.markdown("**Tranche d age**")
        age_range = st.slider(
            "Age (min - max)",
            min_value=15, max_value=40,
            value=(15, 40),
            step=1,
            format="%d ans",
        )

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
            budget_max_eur=budget_max_m * 1_000_000,
            include_unknown_mv=include_unknown_mv,
            age_min=age_range[0],
            age_max=age_range[1],
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
            mv_str    = _fmt_market_value(cand_data.get("market_value_eur"))
            ct_str    = _fmt_contract_end(cand_data.get("contract_end"))
            sal_raw   = _fmt_salary(cand_data.get("annual_gross_eur"))
            estimated = bool(cand_data.get("salary_estimated", False))
            sal_str   = f"~{sal_raw}" if estimated and sal_raw != "N/A" else sal_raw

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
        STYLE_COLORS = {
            "Possession":              "#00d4ff",
            "Pressing / intensite":    "#ff6b35",
            "Jeu direct / physique":   "#7bc67e",
            "Bloc bas / contre-attaque": "#f7b731",
        }
        STYLE_DESC = {
            "Possession":            "Equipe qui conserve le ballon, construit proprement et attend l ouverture. Chercher des joueurs techniques, bons passeurs, confortables sous pression.",
            "Pressing / intensite":  "Equipe qui presse haut, recupere vite et joue vertical. Chercher des joueurs endurants, bons au duel, capable de jouer vite.",
            "Jeu direct / physique": "Equipe qui joue long, s appuie sur la puissance physique et les duels aeriens. Chercher des joueurs physiques, bons de la tete, capables de tenir le ballon.",
            "Bloc bas / contre-attaque": "Equipe qui defend profond et repart en contre. Chercher des joueurs rapides en transition, defenseurs solides, attaquants capables d exploiter l espace.",
        }

        # ── Profil de l equipe recruteuse ─────────────────────────────────────
        team_row2   = teams_df[teams_df["team"] == selected_team]
        X_team2     = prepare_team_features(team_row2)
        cluster_id2 = int(kmeans.predict(X_team2.values)[0])
        style_name2 = label_team_cluster(cluster_id2)
        style_color = STYLE_COLORS.get(style_name2, "#00d4ff")

        st.markdown(
            f'<h2 style="margin-bottom:4px">{selected_team}</h2>'
            f'<span style="background:{style_color}22; color:{style_color}; '
            f'border:1px solid {style_color}; border-radius:6px; '
            f'padding:4px 14px; font-size:1rem; font-weight:700;">'
            f'{style_name2}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="color:#bbb; margin-top:10px; margin-bottom:20px;">'
            f'{STYLE_DESC.get(style_name2, "")}</p>',
            unsafe_allow_html=True,
        )

        # ── Radar : equipe vs moyenne du cluster ──────────────────────────────
        TEAM_RADAR_FEATURES = [
            ("Possession (%)",   "Poss_"),
            ("Tirs/90",          "Standard_Sh/90"),
            ("Buts/90",          "Per 90 Minutes_Gls"),
            ("Pressing",         "fls_per90"),
            ("Interceptions/90", "int_per90"),
            ("Tacles/90",        "tklw_per90"),
            ("Centres/90",       "crs_per90"),
        ]

        all_X2      = prepare_team_features(teams_df)
        all_labels2 = kmeans.predict(all_X2.values)
        same_cluster_mask = all_labels2 == cluster_id2

        radar_labels = [f for _, f in TEAM_RADAR_FEATURES if f in all_X2.columns]
        radar_names  = [n for n, f in TEAM_RADAR_FEATURES if f in all_X2.columns]

        if radar_labels and not team_row2.empty:
            team_vals    = all_X2.loc[team_row2.index, radar_labels].iloc[0].values.astype(float)
            cluster_avg  = all_X2[same_cluster_mask][radar_labels].mean().values.astype(float)
            global_avg   = all_X2[radar_labels].mean().values.astype(float)

            # Normalisation 0-1 sur global max pour garder l echelle comparable
            col_max = np.maximum(all_X2[radar_labels].max().values, 1e-6)
            t_norm  = team_vals   / col_max
            c_norm  = cluster_avg / col_max

            fig_radar = go.Figure()
            for vals, name, color, dash in [
                (t_norm, selected_team,          style_color, "solid"),
                (c_norm, f"Moy. {style_name2}",  "#888888",   "dot"),
            ]:
                v = vals.tolist() + [vals[0]]
                fig_radar.add_trace(go.Scatterpolar(
                    r=v, theta=radar_names + [radar_names[0]],
                    fill="toself" if dash == "solid" else "none",
                    name=name, line=dict(color=color, dash=dash, width=2),
                    opacity=0.85,
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, showticklabels=False, range=[0, 1])),
                showlegend=True,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"), height=400, margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Equipes au style le plus similaire ────────────────────────────────
        st.markdown("### Equipes au style le plus similaire")
        st.caption("Ces equipes jouent comme vous — leurs joueurs s adapteront plus facilement.")

        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
        scaler_team = kmeans.named_steps["scaler"]
        all_X2_scaled = scaler_team.transform(all_X2)
        team_idx_in_all = list(teams_df["team"].values).index(selected_team) if selected_team in teams_df["team"].values else 0
        team_vec   = all_X2_scaled[[team_idx_in_all]]
        sims_teams = _cos_sim(team_vec, all_X2_scaled)[0]

        sim_df = pd.DataFrame({
            "team":       teams_df["team"].values,
            "league":     teams_df["league"].values,
            "similarity": sims_teams,
            "cluster":    all_labels2,
        }).reset_index(drop=True)
        sim_df = sim_df[sim_df["team"] != selected_team]
        sim_df = sim_df.nlargest(9, "similarity")

        cols3 = st.columns(3)
        for i, (_, r) in enumerate(sim_df.iterrows()):
            sname = label_team_cluster(int(r["cluster"]))
            sclr  = STYLE_COLORS.get(sname, "#888")
            sim_p = int(r["similarity"] * 100)
            with cols3[i % 3]:
                st.markdown(
                    f'<div class="metric-card" style="margin-bottom:10px;">'
                    f'<div class="metric-value" style="font-size:1rem; color:#fff">{r["team"]}</div>'
                    f'<div class="metric-label">{r["league"]}</div>'
                    f'<div style="margin-top:6px;">'
                    f'<span style="color:{sclr}; font-size:0.8rem; font-weight:600">{sname}</span>'
                    f'<span style="color:#888; font-size:0.8rem;"> &nbsp;·&nbsp; {sim_p}% similaire</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

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
