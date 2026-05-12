"""
app.py  –  Application Streamlit du Scout Intelligence Tool.

Interface pour :
  1. Analyser le style de jeu d'une équipe
  2. Trouver les joueurs compatibles avec ce style
  3. Filtrer par budget (valeur marchande)
  4. Comparer les profils via radar chart
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import DATA_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402
from features import (  # noqa: E402
    prepare_team_features,
    prepare_player_features,
    get_role_label,
    label_team_cluster,
    PLAYER_ROLE_FEATURES,
)
from model_io import load_model  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Cache data / models
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    players_path = DATA_DIR / "players_raw.csv"
    teams_path   = DATA_DIR / "teams_raw.csv"
    if not players_path.exists() or not teams_path.exists():
        return None, None
    return pd.read_csv(players_path), pd.read_csv(teams_path)


@st.cache_resource
def load_models():
    try:
        kmeans = load_model(MODELS_DIR / "kmeans_team.joblib")
        rf     = load_model(MODELS_DIR / "rf_role.joblib")
        knn    = load_model(MODELS_DIR / "knn_similarity.joblib")
        return kmeans, rf, knn
    except FileNotFoundError:
        return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# Radar chart
# ─────────────────────────────────────────────────────────────────────────────

def radar_chart(profiles: dict[str, pd.Series], features: list[str]) -> go.Figure:
    """Construit un radar chart Plotly comparant plusieurs profils joueurs."""
    fig = go.Figure()
    colors = ["#00d4ff", "#ff6b35", "#7bc67e", "#f7b731", "#a29bfe"]

    for i, (name, profile) in enumerate(profiles.items()):
        vals = [float(profile.get(f, 0)) for f in features]
        vals_norm = []
        for v in vals:
            vals_norm.append(v)
        # Fermer le polygone
        vals_norm.append(vals_norm[0])
        feats_closed = features + [features[0]]

        fig.add_trace(go.Scatterpolar(
            r=vals_norm,
            theta=feats_closed,
            fill="toself",
            name=name,
            line_color=colors[i % len(colors)],
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=450,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> None:
    # ── Page config ──────────────────────────────────────────────────────────
    st.set_page_config(
        page_title="Scout Intelligence Tool",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
        .main-title { font-size: 2.5rem; font-weight: 800; color: #00d4ff; }
        .subtitle   { color: #aaa; margin-top: -10px; margin-bottom: 20px; }
        .metric-card {
            background: #1e1e2e; border-radius: 12px;
            padding: 16px; text-align: center; margin: 4px;
        }
        .metric-value { font-size: 1.8rem; font-weight: 700; color: #00d4ff; }
        .metric-label { font-size: 0.85rem; color: #aaa; }
        .player-card {
            background: #1e1e2e; border-radius: 10px;
            padding: 14px; margin: 6px 0;
            border-left: 3px solid #00d4ff;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="main-title">⚽ Scout Intelligence Tool</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Trouve les joueurs compatibles avec ton style de jeu — '
        'dans les ligues secondaires européennes.</p>',
        unsafe_allow_html=True,
    )

    # ── Load data & models ────────────────────────────────────────────────────
    players_df, teams_df = load_data()
    kmeans, rf, knn      = load_models()

    if players_df is None or teams_df is None:
        st.error("❌ Données manquantes. Lance d'abord : `python scripts/collect_data.py`")
        st.stop()

    if kmeans is None:
        st.error("❌ Modèles manquants. Lance d'abord : `python scripts/train_models.py`")
        st.stop()

    # ── Sidebar — paramètres ──────────────────────────────────────────────────
    with st.sidebar:
        st.header("🔧 Paramètres")

        # Sélection équipe
        team_list = sorted(teams_df["team"].dropna().unique().tolist())
        selected_team = st.selectbox("Équipe de référence (style de jeu)", team_list)

        # Poste recherché
        role_options = {
            "Gardien":          "goalkeeper",
            "Défenseur":        "defender",
            "Milieu défensif":  "defensive_mid",
            "Milieu central":   "central_mid",
            "Milieu offensif":  "attacking_mid",
            "Ailier":           "winger",
            "Attaquant":        "striker",
        }
        selected_role_label = st.selectbox("Poste recherché", list(role_options.keys()))
        selected_role = role_options[selected_role_label]

        # Nombre de recommandations
        n_recommendations = st.slider("Nombre de recommandations", 5, 20, 10)

        # Exclure la ligue de l'équipe de référence (pour chercher ailleurs)
        team_league = teams_df.loc[teams_df["team"] == selected_team, "league"]
        team_league_val = team_league.iloc[0] if len(team_league) > 0 else ""
        exclude_own_league = st.checkbox("Exclure la propre ligue de l'équipe", value=True)

        st.markdown("---")
        st.caption("Données : FBref via soccerdata | Ligues secondaires européennes 2023-24")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Recommandations",
        "🏟️ Style d'équipe",
        "📊 Métriques modèles",
        "ℹ️ À propos",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 1 — Recommandations
    # ═══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.subheader(f"Joueurs recommandés pour {selected_team} — poste : {selected_role_label}")

        # Style de l'équipe sélectionnée
        team_row = teams_df[teams_df["team"] == selected_team]
        if team_row.empty:
            st.warning("Équipe introuvable dans le dataset.")
            st.stop()

        X_team = prepare_team_features(team_row)
        cluster_id = int(kmeans.predict(X_team)[0])
        style_name = label_team_cluster(cluster_id)

        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-card"><div class="metric-value">{style_name}</div>'
                      f'<div class="metric-label">Style de jeu détecté</div></div>',
                      unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><div class="metric-value">{selected_role_label}</div>'
                      f'<div class="metric-label">Poste recherché</div></div>',
                      unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><div class="metric-value">{n_recommendations}</div>'
                      f'<div class="metric-label">Candidats</div></div>',
                      unsafe_allow_html=True)

        st.markdown("---")

        # Filtrer joueurs par rôle prédit
        players_df["predicted_role"] = players_df["position"].apply(get_role_label)
        filtered = players_df[players_df["predicted_role"] == selected_role].copy()

        if exclude_own_league and team_league_val:
            filtered = filtered[filtered["league"] != team_league_val]

        if filtered.empty:
            st.warning("Aucun joueur trouvé pour ce poste dans les ligues sélectionnées.")
            st.stop()

        # Profil moyen de l'équipe pour calculer la compatibilité
        team_player_rows = players_df[players_df["team"] == selected_team]
        if team_player_rows.empty:
            st.info("Pas de données joueurs individuelles pour cette équipe — compatibilité basée sur le style global.")
            team_player_profile = prepare_player_features(filtered).median()
        else:
            team_player_profile = prepare_player_features(team_player_rows).median()

        # Calcul score de compatibilité (similarité cosinus avec le profil moyen de l'équipe)
        X_filtered = prepare_player_features(filtered)
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        team_vec = scaler.transform(team_player_profile.values.reshape(1, -1))

        similarity = cosine_similarity(team_vec, X_scaled)[0]
        filtered = filtered.copy()
        filtered["compatibility_score"] = similarity

        # Top N
        top_players = filtered.nlargest(n_recommendations, "compatibility_score")

        # Affichage
        for _, row in top_players.iterrows():
            score = row["compatibility_score"]
            score_pct = int(score * 100)
            score_color = "#7bc67e" if score_pct >= 80 else "#f7b731" if score_pct >= 60 else "#ff6b35"

            player_name = row.get("player", "Inconnu")
            team_name   = row.get("team",   "?")
            league_name = row.get("league", "?")
            age         = row.get("age",    "?")
            nationality = row.get("nationality", "")

            st.markdown(
                f'<div class="player-card">'
                f'<b style="font-size:1.1rem">{player_name}</b> '
                f'<span style="color:#aaa">({age} ans {nationality})</span><br>'
                f'<span style="color:#aaa">{team_name} — {league_name}</span><br>'
                f'<b style="color:{score_color}">Compatibilité : {score_pct}%</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Radar chart comparaison top 3
        st.markdown("### Comparaison radar — Top 3 candidats")
        top3 = top_players.head(3)
        profiles = {}
        for _, row in top3.iterrows():
            name = row.get("player", "?")
            row_df = pd.DataFrame([row])
            feat_series = prepare_player_features(row_df).iloc[0]
            profiles[name] = feat_series

        # Normaliser pour le radar (0-1)
        all_vals = pd.DataFrame(profiles).T
        mins = all_vals.min()
        maxs = all_vals.max()
        rng  = (maxs - mins).replace(0, 1)
        normalized = (all_vals - mins) / rng

        display_features = [f for f in PLAYER_ROLE_FEATURES if f in normalized.columns][:10]
        norm_profiles = {name: normalized.loc[name] for name in profiles}

        fig = radar_chart(norm_profiles, display_features)
        st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 2 — Style d'équipe
    # ═══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.subheader("🏟️ Analyse du style de jeu par équipe")

        all_teams = sorted(teams_df["team"].dropna().unique())
        compare_teams = st.multiselect(
            "Sélectionne 2 à 4 équipes à comparer",
            all_teams,
            default=[selected_team] + [t for t in all_teams if t != selected_team][:2],
            max_selections=4,
        )

        if len(compare_teams) < 2:
            st.info("Sélectionne au moins 2 équipes.")
        else:
            rows = teams_df[teams_df["team"].isin(compare_teams)]
            X_compare = prepare_team_features(rows)
            cluster_ids = kmeans.predict(X_compare)

            cols = st.columns(len(compare_teams))
            for i, (team, cid) in enumerate(zip(compare_teams, cluster_ids)):
                with cols[i]:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value" style="font-size:1.2rem">{team}</div>'
                        f'<div class="metric-label">{label_team_cluster(int(cid))}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Distribution des styles dans tout le dataset
            st.markdown("### Distribution des styles de jeu")
            all_X = prepare_team_features(teams_df)
            all_clusters = kmeans.predict(all_X)
            cluster_counts = pd.Series(all_clusters).value_counts().sort_index()
            cluster_labels = [label_team_cluster(i) for i in cluster_counts.index]

            fig2 = go.Figure(go.Bar(
                x=cluster_labels,
                y=cluster_counts.values,
                marker_color=["#00d4ff", "#ff6b35", "#7bc67e", "#f7b731"],
            ))
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                yaxis_title="Nombre d'équipes",
                height=350,
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 3 — Métriques modèles
    # ═══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.subheader("📊 Performance des modèles")

        metrics_path = RESULTS_DIR / "model_metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            st.dataframe(metrics_df, use_container_width=True)

            # Bar chart métriques
            numeric_cols = metrics_df.select_dtypes(include="number").columns.tolist()
            if numeric_cols and "model_name" in metrics_df.columns:
                fig3 = go.Figure()
                colors = ["#00d4ff", "#ff6b35", "#7bc67e", "#f7b731", "#a29bfe", "#fd79a8"]
                for i, col in enumerate(numeric_cols[:6]):
                    fig3.add_trace(go.Bar(
                        name=col,
                        x=metrics_df["model_name"],
                        y=metrics_df[col],
                        marker_color=colors[i % len(colors)],
                    ))
                fig3.update_layout(
                    barmode="group",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="white"),
                    height=400,
                )
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Lance `python scripts/main.py` pour générer les métriques.")

    # ═══════════════════════════════════════════════════════════════════════════
    # TAB 4 — À propos
    # ═══════════════════════════════════════════════════════════════════════════
    with tab4:
        st.subheader("ℹ️ À propos du projet")
        st.markdown("""
        ### Scout Intelligence Tool

        **Problème résolu :** Les clubs à petit budget n'ont pas les moyens de payer
        des plateformes comme Wyscout (50 000€+/an). Ils font leurs recrutements
        manuellement, au feeling, en regardant des matchs.

        **Notre approche :** Un outil data-driven qui, à partir du style de jeu
        d'une équipe, identifie les joueurs des ligues secondaires européennes
        dont le profil tactique est le plus compatible — pour un coût accessible.

        ---

        ### Pipeline technique

        1. **KMeans (clustering)** — Classe chaque équipe selon son style de jeu
           (possession, pressing, jeu direct, bloc bas)
        2. **Random Forest (classification)** — Identifie le vrai rôle tactique
           d'un joueur depuis ses statistiques
        3. **KNN + similarité cosinus** — Mesure la compatibilité entre un profil
           joueur et le système de jeu d'une équipe

        ---

        ### Données
        - **Source :** FBref via la librairie open-source `soccerdata`
        - **Ligues :** Championship, Ligue 2, 2. Bundesliga, Serie B,
          Eredivisie, Pro League belge, Primeira Liga, Segunda División
        - **Saison :** 2023-24
        - **Filtrage :** Joueurs avec ≥ 500 minutes jouées

        ---

        ### Limites honnêtes
        > La data ne remplace pas l'œil du scout. Elle lui fait gagner du temps
        > en réduisant le vivier de 2000 joueurs à 10 candidats pertinents.
        > Le scout garde toujours le dernier mot.
        """)


if __name__ == "__main__":
    build_app()
