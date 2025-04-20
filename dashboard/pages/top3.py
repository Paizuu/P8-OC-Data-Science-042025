# 📄 top3.py — Variables extrêmes (z-score)
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Top/Flop 3 variables", layout="centered")
st.title("Variables extrêmes du client")

st.markdown("Cette section met en évidence les 3 variables où le client se situe aux extrémités faibles ou élevées par rapport à la population. Cela permet d’identifier rapidement les caractéristiques qui le distinguent fortement.")

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

# Récupération client
if "selected_client_id" not in st.session_state:
    st.warning("Veuillez d'abord sélectionner un identifiant client depuis la page principale.")
    st.stop()

client_id = st.session_state["selected_client_id"]
if client_id not in data.index:
    st.error("Client introuvable dans les données.")
    st.stop()

client_data = data.loc[client_id]


# Sélection des colonnes pertinentes
numeric_data = data.select_dtypes(include=["int64", "float64"]).copy()
numeric_data = numeric_data.loc[:, numeric_data.nunique() >= 3]
client_numeric = client_data[numeric_data.columns]

# Calcul des z-scores
z_scores = {}
for col in numeric_data.columns:
    population = numeric_data[col].dropna()
    mean = population.mean()
    std = population.std()
    client_val = client_numeric[col]
    if std > 0:
        z = (client_val - mean) / std
        z_scores[col] = z

z_df = pd.DataFrame.from_dict(z_scores, orient="index", columns=["z"])
z_df["abs_z"] = z_df["z"].abs()

low_features = z_df[z_df["z"] < 0].sort_values("z").head(3).index.tolist()
high_features = z_df[z_df["z"] > 0].sort_values("z", ascending=False).head(3).index.tolist()

st.markdown("### Variables faibles du client")
st.markdown("* Top 3 des caractéristiques où le client présente les valeurs les plus faibles par rapport à l’ensemble de la population.")
for feature in low_features:
    val = client_data[feature]
    pop = data[feature].dropna()
    fig, ax = plt.subplots()
    ax.hist(pop, bins=30, color="lightgray", edgecolor="black")
    ax.axvline(val, color="red", linestyle="--", label=f"Client ({val:.2f})")
    ax.set_title(f"{feature}")
    ax.legend()
    st.pyplot(fig)


st.markdown("### Variables fortes du client")
st.markdown("* Top 3 des caractéristiques où le client présente les valeurs les plus élevées par rapport à l’ensemble de la population.")

for feature in high_features:
    val = client_data[feature]
    pop = data[feature].dropna()
    fig, ax = plt.subplots()
    ax.hist(pop, bins=30, color="lightgray", edgecolor="black")
    ax.axvline(val, color="red", linestyle="--", label=f"Client ({val:.2f})")
    ax.set_title(f"{feature}")
    ax.legend()
    st.pyplot(fig)
