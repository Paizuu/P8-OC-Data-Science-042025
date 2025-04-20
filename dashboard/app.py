import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import requests 


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

# Stockage de l'ID client dans le session_state
# Définir les bornes min/max
min_id = int(data.index.min())
max_id = int(data.index.max())

# Lire la valeur stockée dans session_state, ou prendre la valeur minimale si elle n'existe pas encore
default_id = st.session_state.get("selected_client_id", min_id)

# Créer le champ avec cette valeur
client_id = st.number_input(
    "Entrez l'identifiant client",
    min_value=min_id,
    max_value=max_id,
    step=1,
    value=default_id
)

# Stocker la nouvelle valeur si modifiée
st.session_state["selected_client_id"] = client_id

if client_id not in data.index:
    st.warning("Veuillez saisir un ID valide.")
    st.stop()

client_data = data.loc[client_id]
example_data = client_data.to_dict()

# Affichage des données du client
st.markdown("### Données du client sélectionné")
st.dataframe(pd.DataFrame(example_data.items(), columns=["Champ", "Valeur"]))

# Chargement du modèle local
@st.cache_resource
def load_model():
    return joblib.load("dashboard/model/model.pkl")

pipeline = load_model()

# # Prédiction locale
# st.write("Cliquez ci-dessous pour prédire le score d’un client à partir des données actuelles.")
# if st.button("Prédire"):
#     try:
#         input_df = pd.DataFrame([example_data])
#         proba = pipeline.predict_proba(input_df)[0]
#         pred_class = pipeline.predict(input_df)[0]

#         score = proba[0] * 100  # TARGET 0 = remboursé
#         antiscore = proba[1] * 100  # TARGET 1 = défaut
#         decision = "ACCORDÉ" if pred_class == 0 else "REFUSÉ"

#         st.subheader(f"Décision : {decision}")
#         st.write(f"Score de remboursement : {score:.2f} %")
#         st.write(f"Score de défaut : {antiscore:.2f} %")

#         fig = go.Figure(go.Indicator(
#             mode="gauge+number+delta",
#             value=score,
#             delta={'reference': 35},
#             gauge={
#                 'axis': {'range': [0, 100]},
#                 'steps': [
#                     {'range': [0, 35], 'color': "red"},
#                     {'range': [35, 100], 'color': "lightgreen"}
#                 ],
#                 'threshold': {'line': {'color': "black", 'width': 4}, 'value': 35}
#             },
#             title={'text': "Score de remboursement (%)"}
#         ))
#         st.plotly_chart(fig)

#     except Exception as e:
#         st.error(f"Erreur lors de la prédiction : {e}")



# Alternative : appel API externe (commenté)
API_URL = "https://mdn-antoine-projet7-implementez-un.onrender.com/predict"

# Set de la fonction de prédiction via API 
st.write("Cliquez ci-dessous pour prédire le score d’un client à partir d’un exemple de données.")
if st.button("Prédire"):
    try:
        response = requests.post(API_URL, json=example_data)
        if response.status_code == 200:
            result = response.json()
            score = result["Prédiction de la TARGET 0"] * 100
            antiscore = result["Prédiction de la TARGET 1"] * 100
            decision = "ACCORDÉ" if result["Classe prédite pour ces données"] == 0 else "REFUSÉ"

            st.subheader(f"Décision : {decision}")
            st.write(f"Score de remboursement : {score:.2f} %")
            st.write(f"Score de remboursement : {antiscore:.2f} %")

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=score,
                delta={'reference': 35},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 35], 'color': "red"},
                        {'range': [35, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'value': 35}
                },
                title={'text': "Score de remboursement (%)"}
            ))
            st.plotly_chart(fig)
        else:
            st.error("Erreur API : " + response.text)
    except Exception as e:
        st.error(f"Erreur : {e}")