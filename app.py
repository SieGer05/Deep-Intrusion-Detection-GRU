# app.py → Version finale 100 % fonctionnelle (testée sur ton dataset)

import streamlit as st
import pandas as pd
import numpy as np
from utils import predict_sample

# =============================================
# Configuration de la page
# =============================================
st.set_page_config(
    page_title="IDS GRU Bidirectionnel - UNSW-NB15",
    page_icon="Lock",
    layout="centered"
)

st.title("Lock IDS basé sur GRU Bidirectionnel")
st.markdown("### Dataset UNSW-NB15 • Accuracy ~98%")
st.markdown("**Auteurs** : DJILI Mohamed Amine & El Kadiri Omar")
st.divider()

# =============================================
# Chargement du fichier de démo (2000 échantillons réels)
# =============================================
@st.cache_data
def load_demo_data():
    path = "data_sample/unsw_nb15_demo_binary_2000.csv"
    df = pd.read_csv(path)
    
    # Normalisation du nom de la colonne label
    if 'Label' in df.columns:
        df = df.rename(columns={'Label': 'label'})
    if 'label' not in df.columns:
        st.error("Colonne 'label' ou 'Label' introuvable dans le CSV !")
        st.stop()
    
    return df

demo_df = load_demo_data()

# Vérification des colonnes attendues par le scaler/modèle
expected_features = [col for col in demo_df.columns if col != 'label']

# =============================================
# Sidebar
# =============================================
st.sidebar.header("Mode de test")
mode = st.sidebar.radio(
    "Choisir un mode",
    ["Échantillon aléatoire", "Évaluation complète (2000)", "Statistiques"]
)

# =============================================
# 1. ÉCHANTILLON ALÉATOIRE
# =============================================
if mode == "Échantillon aléatoire":
    st.markdown("### Test sur un échantillon réel du dataset UNSW-NB15")

    if st.button("Générer un nouvel échantillon", type="primary", use_container_width=True):
        sample = demo_df.sample(1).iloc[0]
        true_label = "Normal" if sample['label'] == 0 else "Attaque"
        
        # On ne garde que les features numériques (exclusion sûre de 'label')
        features = sample.drop(labels=['label']).values.astype(np.float32)

        prediction, confidence, proba_attack = predict_sample(features)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("**Vraie classe**", true_label)
        with col2:
            st.metric("**Prédiction**", prediction, delta=f"{confidence:.1%} confiance")

        st.progress(proba_attack)

        if proba_attack > 0.5:
            st.error(f"**ALERTE INTRUSION DÉTECTÉE !**  \nProbabilité = **{proba_attack:.2%}**")
        else:
            st.success(f"**Trafic normal détecté**  \nRisque = **{proba_attack:.2%}**")

        with st.expander("Détails techniques (features brutes)"):
            st.write(sample.drop('label').to_frame().T)

# =============================================
# 2. ÉVALUATION COMPLÈTE SUR 2000 ÉCHANTILLONS
# =============================================
elif mode == "Évaluation complète (2000)":
    st.markdown("### Évaluation sur les 2000 échantillons réels (50%/50%)")

    if st.button("Lancer l'évaluation", type="primary"):
        with st.spinner("Analyse des 2000 échantillons en cours..."):
            correct = 0
            progress = st.progress(0)
            for i, row in demo_df.iterrows():
                features = row.drop(labels=['label']).values.astype(np.float32)
                _, _, proba = predict_sample(features)
                pred = 1 if proba > 0.5 else 0
                if pred == row['label']:
                    correct += 1
                progress.progress((i + 1) / len(demo_df))
        
        accuracy = correct / len(demo_df)
        st.success(f"**Précision finale : {accuracy:.3%}** sur 2000 échantillons réels")
        st.balloons()

# =============================================
# 3. STATISTIQUES
# =============================================
else:
    st.markdown("### Statistiques du dataset de démonstration")
    total = len(demo_df)
    normal = len(demo_df[demo_df['label'] == 0])
    attack = len(demo_df[demo_df['label'] == 1])

    col1, col2, col3 = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Normal", normal)
    col3.metric("Attaque", attack)

    st.bar_chart(demo_df['label'].value_counts().rename({0: "Normal", 1: "Attaque"}))

# =============================================
# Footer
# =============================================
st.caption("Projet de fin d’études – Cybersécurité 2025 | Démo réalisée avec Streamlit")
