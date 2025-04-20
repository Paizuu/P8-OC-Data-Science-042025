import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="SHAP Feature Importances", layout="wide")
st.title(" Analyse des contributions des variables avec SHAP")

st.markdown("Cette section montre quelles variables influencent le plus la décision du modèle")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("ProcessedData/app_test_domain.csv")
    df.drop(columns="SK_ID_CURR", inplace=True)
    df.dropna(inplace=True)
    df = df.rename(columns={
        'NAME_EDUCATION_TYPE_Secondary / secondary special': 'NAME_EDUCATION_TYPE_Secondary_special',
        'NAME_FAMILY_STATUS_Single / not married': 'NAME_FAMILY_STATUS_Single_not_married',
        'NAME_HOUSING_TYPE_House / apartment': 'NAME_HOUSING_TYPE_House_apartment',
        'OCCUPATION_TYPE_Waiters/barmen staff': 'OCCUPATION_TYPE_Waiters_barmen_staff',
        'WALLSMATERIAL_MODE_Stone, brick': 'WALLSMATERIAL_MODE_Stone_brick'
    })
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)
    df.reset_index(drop=True, inplace=True)
    return df



# Chargement du modèle
@st.cache_resource
def load_model():
    return joblib.load("dashboard/model/model.pkl")


# Récupération des données (pipeline, model et ID client + Données globales)
pipeline = load_model()
model = pipeline.named_steps["rff"]
client_id = st.session_state["selected_client_id"]
data = load_data()
X = data.astype("float64")

# Sélection du client
if "selected_client_id" not in st.session_state:
    st.warning("Veuillez d'abord sélectionner un client dans la page principale.")
    st.stop()


# Explainer SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


# Gestion de la classification binaire ou multi-classe
if isinstance(shap_values, list):  
    # Prendre la classe positive si on est en classification binaire
    shap_values_class = shap_values[1] if len(shap_values) > 1 else shap_values[0]
else:
    shap_values_class = shap_values

shap_importance = np.abs(shap_values_class).mean(axis=0)
shap_importance_mean = shap_importance.mean(axis=1) 

# Création du DataFrame des importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'SHAP Importance': shap_importance_mean
}).sort_values(by='SHAP Importance', ascending=False)


# Feature importance Globale via SHAP
st.subheader("Feature importances (top20) de l’ensemble des clients.")
st.markdown("Visualisation des variables les plus influentes dans les décisions du modèle sur l’ensemble des clients.")
top_n = 20
top_features = feature_importance_df.head(top_n)["Feature"].values 
X_top = X[top_features] 

feature_indices = [list(X.columns).index(feature) for feature in top_features]  
shap_values_top = shap_values_class[:, feature_indices]  

if shap_values_top.ndim == 3: 
    shap_values_top = shap_values_top.mean(axis=2)

# Affichage en summary plot des 20 features les plus importantes
fig_summary, ax = plt.subplots()
shap.summary_plot(shap_values_top, X_top, max_display=top_n)
st.pyplot(fig_summary)


# Feature importance Local via SHAP du Client ID

st.subheader("Feature importances du client selectionné")
st.markdown(f"Affichage des variables qui ont le plus contribué à la prédiction pour le client {client_id}.")

sample = X.iloc[[client_id]]  # Données du client 

# SHAP values du client
shap_values_sample = shap_values_class[client_id, :] 

if shap_values_sample.ndim == 2:  # Cas multi-classe
    shap_values_sample = shap_values_sample[:, 1]  # On prend la classe positive

# Correction de base_values pour éviter l'erreur TypeError
base_value = explainer.expected_value
if isinstance(base_value, np.ndarray) or isinstance(base_value, list):  
    base_value = base_value[1] if len(base_value) > 1 else base_value[0] 
base_value = float(np.squeeze(base_value))


# Explication SHAP
explanation = shap.Explanation(
    values=shap_values_sample, 
    base_values=base_value,  # Correction ici
    data=sample.iloc[0]
)

# Affichage via streamlit
plt.clf()
shap.plots.waterfall(explanation, show=False)
st.pyplot(plt.gcf())