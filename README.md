# ⚽ Scout Intelligence Tool

> Trouve les joueurs compatibles avec ton style de jeu dans les ligues secondaires européennes.

Projet ML — Template étudiant adapté par [Ton nom]

---

## Concept

Les clubs à petit budget n'ont pas les moyens de payer des plateformes comme Wyscout.
Cet outil répond à la question :

> **"Étant donné le style de jeu de mon équipe, quels joueurs disponibles s'y intégreraient le mieux ?"**

### Pipeline ML

| Modèle | Rôle |
|--------|------|
| **KMeans** | Classifier le style de jeu d'une équipe |
| **Random Forest** | Identifier le rôle tactique réel d'un joueur |
| **KNN + cosinus** | Mesurer la compatibilité joueur ↔ équipe |

---

## Installation

```bash
# 1. Cloner / forker le repo
git clone <url>
cd football_scout

# 2. Environnement virtuel
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Dépendances
pip install -r requirements.txt
```

---

## Workflow complet

### Étape 1 — Collecter les données
```bash
python scripts/collect_data.py
```
Télécharge les stats joueurs et équipes depuis FBref via `soccerdata`.
Génère `data/players_raw.csv` et `data/teams_raw.csv`.

### Étape 2 — Entraîner les modèles
```bash
python scripts/train_models.py
```
Entraîne les 3 modèles et les sauvegarde dans `models/`.

### Étape 3 — Évaluer et lancer l'app
```bash
python scripts/main.py
```
Évalue les modèles, génère `results/model_metrics.csv`, lance Streamlit sur http://localhost:8501

---

## Structure

```
football_scout/
├── data/                   # Données brutes et traitées
├── models/                 # Modèles entraînés (.joblib)
├── results/                # Métriques d'évaluation
├── plots/                  # Visualisations exportées
├── scripts/
│   ├── collect_data.py     # Téléchargement données FBref
│   ├── train_models.py     # Entraînement des modèles
│   └── main.py             # Orchestration + lancement app
├── src/
│   ├── config.py           # Chemins, paramètres, registre modèles
│   ├── features.py         # Feature engineering (partagé)
│   ├── data.py             # load_dataset_split()
│   ├── metrics.py          # compute_metrics()
│   ├── model_io.py         # load_model()
│   ├── results.py          # write_metrics()
│   └── app.py              # Application Streamlit
└── requirements.txt
```

---

## Données

- **Source :** FBref via [`soccerdata`](https://soccerdata.readthedocs.io/)
- **Ligues :** Championship, Ligue 2, 2. Bundesliga, Serie B, Eredivisie, Pro League belge, Primeira Liga, Segunda División
- **Saison :** 2023-24
- **Filtrage :** ≥ 500 minutes jouées

---

## Limites honnêtes

La data ne remplace pas l'œil du scout. Elle réduit un vivier de 2000 joueurs
à 10 candidats pertinents. Le scout garde le dernier mot.
