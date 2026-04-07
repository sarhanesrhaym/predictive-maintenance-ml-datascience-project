# Projet : Maintenance Prédictive ML

## Contexte
Projet ML end-to-end pour portfolio étudiant (Master 1 IA & IoT).
Objectif : décrocher un stage en ML/Data.

## Problème
Prédire les pannes de machines industrielles avant qu'elles surviennent.
Dataset : AI4I 2020 Predictive Maintenance (UCI) — 10 000 lignes, 14 colonnes.

## Stack technique
- Python 3.x + venv
- pandas, scikit-learn, xgboost, shap
- streamlit (dashboard), fastapi (API)

## Structure du projet
predictive-maintenance-ml/
├── data/           → dataset AI4I
├── notebooks/      → EDA, preprocessing, modélisation
├── src/            → scripts train.py, predict.py
├── app/            → streamlit_app.py
└── models/         → modèles sauvegardés (.pkl)

## Étapes du projet
1. ✅ Mise en place environnement
2. 🔄 EDA (en cours)
3. ⬜ Preprocessing & modélisation
4. ⬜ Explicabilité SHAP
5. ⬜ Dashboard Streamlit + déploiement

## Décisions techniques
- Métrique principale : F1-score + ROC-AUC (pas accuracy, dataset déséquilibré 96.6% / 3.4%)
- Gestion déséquilibre : scale_pos_weight dans XGBoost
- Différenciateur : SHAP pour explicabilité métier