# predictive-maintenance-ml-datascience-project
Prédire les pannes de machines avant qu'elles surviennent, réduire les arrêts non planifiés
# Predictive Maintenance ML — AI4I 2020

> Projet ML end-to-end de maintenance prédictive industrielle.  
> Stack : XGBoost · SHAP · Streamlit · FastAPI · Python

---

## Aperçu

Ce projet prédit les pannes de machines industrielles **avant qu'elles surviennent**,
en analysant les données de capteurs en temps réel (température, couple, vitesse, usure).

Un dashboard interactif permet de simuler les paramètres d'une machine et d'obtenir
instantanément un score de risque avec une explication SHAP de la prédiction.

---

## Démonstration

🔗 **Dashboard live** : [predictive-maintenance.streamlit.app](#) ← remplace par ton URL

---

## Résultats du modèle

| Métrique | Score |
|---|---|
| F1-score (pannes) | 0.73 |
| ROC-AUC | 0.9726 |
| Recall (pannes) | 79% |
| Pannes détectées | 54 / 68 |

> L'accuracy seule n'est pas pertinente ici (dataset déséquilibré : 3.4% de pannes).  
> Les métriques prioritaires sont le F1-score et le ROC-AUC.

---

## Dataset

- **Source** : [AI4I 2020 Predictive Maintenance Dataset — UCI](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **Taille** : 10 000 lignes · 14 colonnes
- **Taux de panne** : 3.4% (dataset déséquilibré)
- **Features** : température air/process, vitesse rotation, couple, usure outil, type machine

---

## Architecture du projet
predictive-maintenance-ml/
├── data/                         → dataset AI4I 2020
├── notebooks/
│   ├── 01_eda.ipynb              → Exploration des données
│   ├── 02_preprocessing_modeling.ipynb  → Modélisation
│   └── 03_shap.ipynb            → Explicabilité SHAP
├── models/
│   ├── xgb_model.pkl            → Modèle XGBoost final
│   ├── shap_explainer.pkl       → Explainer SHAP
│   └── label_encoder.pkl        → Encodeur Type machine
├── app/
│   ├── streamlit_app.py         → Dashboard interactif
│   └── requirements.txt         → Dépendances Streamlit Cloud
├── CLAUDE.md                    → Contexte projet pour Claude AI
├── requirements.txt             → Dépendances locales
└── README.md

---

## Pipeline ML
Données brutes (CSV)
↓
Exploration (EDA)
· Distribution des pannes
· Corrélations entre features
· Types de pannes (HDF, OSF, PWF, TWF, RNF)
↓
Preprocessing
· Encodage LabelEncoder (Type L/M/H)
· Gestion déséquilibre : scale_pos_weight=28.52
· Split stratifié 80/20
↓
Modélisation (comparaison 3 modèles)
· Logistic Regression  → F1: 0.24 · AUC: 0.9062
· Random Forest        → F1: 0.68 · AUC: 0.9699
· XGBoost ✅           → F1: 0.73 · AUC: 0.9726
↓
Explicabilité SHAP
· Beeswarm plot global
· Waterfall plot individuel
↓
Dashboard Streamlit
· Prédiction en temps réel
· Explication SHAP interactive

---

## Insights clés (SHAP)

Les features les plus importantes pour prédire une panne :

1. **Torque (Nm)** — facteur dominant · couple élevé = risque critique
2. **Tool wear (min)** — usure avancée = signal d'alerte prioritaire
3. **Rotational speed (rpm)** — vitesse faible = risque accru
4. **Air temperature (K)** — impact modéré sur le risque

> *"Un couple mécanique anormalement élevé combiné à une usure avancée  
> de l'outil caractérise la majorité des pannes de type HDF et OSF."*

---

## Installation locale

```bash
# 1. Cloner le repo
git clone https://github.com/sarhanesrhaym/predictive-maintenance-ml-datascience-project.git
cd predictive-maintenance-ml-datascience-project

# 2. Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer le dashboard
cd app
streamlit run streamlit_app.py
```

---

## Stack technique

| Catégorie | Technologies |
|---|---|
| Langage | Python 3.x |
| ML | scikit-learn · XGBoost |
| Explicabilité | SHAP |
| Dashboard | Streamlit |
| Data | Pandas · NumPy |
| Visualisation | Matplotlib · Seaborn |
| Versioning | Git · GitHub |
| Déploiement | Streamlit Cloud |

---

## Auteur

**Sarhane Srhaym**  
Étudiant Master 1 IA & Objets Connectés — Université Ibn Tofail, Kénitra  
En recherche de stage en ML / Data Science

[![GitHub](https://img.shields.io/badge/GitHub-sarhanesrhaym-181717?logo=github)](https://github.com/sarhanesrhaym)