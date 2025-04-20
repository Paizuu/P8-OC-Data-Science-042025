import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Comparaison client", layout="centered")
st.title("Comparaison des variables client avec celles de la population")

st.markdown("Ce graphique montre la distribution d'une variable dans l'ensemble de la population. La ligne rouge indique la valeur du client, ce qui permet de voir s'il est plutôt dans la moyenne ou s'il se démarque.")

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

if "selected_client_id" not in st.session_state:
    st.warning("Veuillez d'abord sélectionner un identifiant client depuis la page principale.")
    st.stop()

client_id = st.session_state["selected_client_id"]
if client_id not in data.index:
    st.error("L'identifiant client sélectionné n'existe pas dans les données.")
    st.stop()

client_data = data.loc[client_id]

# Choisir la variable à comparer
selected_feature = st.selectbox(
    "Sélectionnez une variable à comparer",
    options=data.columns.sort_values()
)

valeur_client = client_data[selected_feature]
valeurs_population = data[selected_feature]

# Vérifie que la conversion a fonctionné (pas bool, pas object)
if pd.api.types.is_numeric_dtype(valeurs_population):
    valeurs_population = valeurs_population.dropna()
    fig, ax = plt.subplots()
    ax.hist(valeurs_population, bins=30, color="lightgray", edgecolor="black")
    if pd.notnull(valeur_client):
        ax.axvline(valeur_client, color="red", linestyle="--", label=f"Client ({valeur_client:.2f})")
        ax.legend()
    ax.set_title(f"Distribution de '{selected_feature}'")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Nombre de clients")
    st.pyplot(fig)

elif pd.api.types.is_bool_dtype(valeurs_population) or valeurs_population.nunique() <= 2:
    counts = valeurs_population.value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values, color="lightgray", edgecolor="black")
    ax.set_title(f"Répartition de '{selected_feature}'")
    ax.set_ylabel("Nombre de clients")
    client_val_str = str(int(valeur_client))
    if client_val_str in counts.index.astype(str).tolist():
        ax.bar(client_val_str, counts.loc[int(client_val_str)], color="red")
    st.pyplot(fig)
else:
    st.warning(f"'{selected_feature}' n'est pas une variable numérique. Impossible de générer un histogramme.")
