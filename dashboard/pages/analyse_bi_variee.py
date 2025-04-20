import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Analyse bi-variée", layout="centered")
st.title("Analyse bi-variée entre deux variables")

st.markdown("Ce graphique montre la relation entre deux variables pour tous les clients. Le point rouge correspond au client sélectionné. Cela permet de voir si ses données sont cohérentes ou atypiques par rapport à la population générale.")

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv("ProcessedData/app_train_domain.csv")
    df = df.drop(columns='SK_ID_CURR')
    df = df.dropna()
    df = df.rename(columns={
        'NAME_EDUCATION_TYPE_Secondary / secondary special': 'NAME_EDUCATION_TYPE_Secondary_special',
        'NAME_FAMILY_STATUS_Single / not married': 'NAME_FAMILY_STATUS_Single_not_married',
        'NAME_HOUSING_TYPE_House / apartment': 'NAME_HOUSING_TYPE_House_apartment',
        'OCCUPATION_TYPE_Waiters/barmen staff': 'OCCUPATION_TYPE_Waiters_barmen_staff',
        'WALLSMATERIAL_MODE_Stone, brick': 'WALLSMATERIAL_MODE_Stone_brick'
    })
    df.reset_index(drop=True, inplace=True)
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)
    return df

data = load_data()

# Récupération de l'ID client
if "selected_client_id" not in st.session_state:
    st.warning("Veuillez d'abord sélectionner un identifiant client depuis la page principale.")
    st.stop()

client_id = st.session_state["selected_client_id"]
if client_id not in data.index:
    st.error("L'identifiant client sélectionné n'existe pas dans les données.")
    st.stop()

client_data = data.loc[client_id]

# Sélection des deux variables
num_cols = data.select_dtypes(include=["int64", "float64"]).columns
col1, col2 = st.columns(2)

with col1:
    x_var = st.selectbox("Variable en abscisse (X)", options=sorted(num_cols), key="x_var")
with col2:
    y_var = st.selectbox("Variable en ordonnée (Y)", options=sorted(num_cols), key="y_var")

# Affichage du scatterplot
fig, ax = plt.subplots()
ax.scatter(data[x_var], data[y_var], alpha=0.3, label="Population")
ax.scatter(client_data[x_var], client_data[y_var], color="red", label="Client", edgecolor="black", s=100)
ax.set_xlabel(x_var)
ax.set_ylabel(y_var)
ax.set_title(f"Comparaison bi-variée : {x_var} vs {y_var}")
ax.legend()
st.pyplot(fig)
