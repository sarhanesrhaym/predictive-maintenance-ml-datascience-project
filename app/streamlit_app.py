import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

try:
    import shap
except ModuleNotFoundError:
    shap = None

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

@st.cache_resource
def load_models():
    xgb_path = MODELS_DIR / "xgb_model.pkl"
    explainer_path = MODELS_DIR / "shap_explainer.pkl"

    if not xgb_path.exists():
        missing = []
        if not xgb_path.exists():
            missing.append(str(xgb_path))
        raise FileNotFoundError("Fichiers modèles introuvables: " + ", ".join(missing))

    xgb = joblib.load(xgb_path)
    explainer = None

    if shap is not None and explainer_path.exists():
        explainer = joblib.load(explainer_path)

    return xgb, explainer

try:
    xgb, explainer = load_models()
except Exception as e:
    st.error(f"Erreur au chargement des modèles : {e}")
    st.info("Vérifie que le dossier `models/` existe à la racine du projet.")
    st.stop()

st.title("Predictive Maintenance Dashboard")
st.caption("Prédiction de panne machine en temps réel — AI4I 2020")

st.sidebar.header("Paramètres machine")

air_temp = st.sidebar.slider(
    'Température air (K)', 295.0, 305.0, 300.0, 0.1)
proc_temp = st.sidebar.slider(
    'Température process (K)', 305.0, 315.0, 310.0, 0.1)
rpm = st.sidebar.slider(
    'Vitesse rotation (rpm)', 1168, 2886, 1500)
torque = st.sidebar.slider(
    'Couple (Nm)', 3.0, 77.0, 40.0, 0.1)
wear = st.sidebar.slider(
    'Usure outil (min)', 0, 253, 100)
machine_type = st.sidebar.selectbox(
    'Type machine', ['L', 'M', 'H'])

type_enc = {'L': 0, 'M': 1, 'H': 2}[machine_type]

features = [
    'Air_temperature_K',
    'Process_temperature_K',
    'Rotational_speed_rpm',
    'Torque_Nm',
    'Tool_wear_min',
    'Type_encoded'
]

X_input = pd.DataFrame(
    [[air_temp, proc_temp, rpm, torque, wear, type_enc]],
    columns=features
)

proba = xgb.predict_proba(X_input)[0][1]
risk_pct = round(proba * 100)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Risque de panne", f"{risk_pct}%")
with col2:
    status = "CRITIQUE" if risk_pct > 60 else "ATTENTION" if risk_pct > 30 else "NORMAL"
    st.metric("Statut", status)
with col3:
    st.metric("Couple (Nm)", f"{torque}")

if risk_pct > 60:
    st.error(f"Risque critique : {risk_pct}% — intervention recommandée")
elif risk_pct > 30:
    st.warning(f"Risque modéré : {risk_pct}% — surveillance accrue")
else:
    st.success(f"Machine en bon état : {risk_pct}% de risque")

st.progress(risk_pct)

st.subheader("Explication de la prédiction (SHAP)")

if shap is None:
    st.warning(
        "SHAP n'est pas installé dans cet environnement Python. "
        "Installe-le puis relance l'app pour afficher l'explication SHAP."
    )
elif explainer is None:
    st.warning("Le fichier `models/shap_explainer.pkl` est introuvable ou non chargé.")
else:
    try:
        shap_vals = explainer.shap_values(X_input)

        fig, ax = plt.subplots(figsize=(8, 3))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=X_input.iloc[0].values,
                feature_names=features
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.error(f"Erreur lors de l'affichage SHAP : {e}")

st.subheader("Valeurs des capteurs")
st.dataframe(X_input.rename(columns={
    'Air_temperature_K'     : 'Temp. air (K)',
    'Process_temperature_K' : 'Temp. process (K)',
    'Rotational_speed_rpm'  : 'Vitesse (rpm)',
    'Torque_Nm'             : 'Couple (Nm)',
    'Tool_wear_min'         : 'Usure outil (min)',
    'Type_encoded'          : 'Type machine'
}), use_container_width=True)