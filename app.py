import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Configuration de la page
st.set_page_config(
    page_title="IDS - Détection d'Intrusions avec GRU",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .attack-alert {
        background-color: #ff4444;
        padding: 1rem;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    .normal-alert {
        background-color: #00C851;
        padding: 1rem;
        border-radius: 5px;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Titre principal
st.markdown('<h1 class="main-header">🛡️ Système de Détection d\'Intrusions (IDS)</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Détection en temps réel avec Deep Learning (GRU Bidirectionnel)</p>', unsafe_allow_html=True)

# Chargement des modèles et preprocesseurs
@st.cache_resource
def load_models():
    try:
        model = load_model('models/ids_gru_model.keras')
        scaler = joblib.load('models/scaler_std.pkl')
        encoders = joblib.load('models/label_encoders.pkl')
        dropped_cols = joblib.load('models/dropped_columns.pkl')
        
        # Récupérer les noms de colonnes attendus par le scaler
        if hasattr(scaler, 'feature_names_in_'):
            expected_features = list(scaler.feature_names_in_)
        else:
            expected_features = None
            
        return model, scaler, encoders, dropped_cols, expected_features
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        return None, None, None, None, None

model, scaler, encoders, dropped_cols, expected_features = load_models()

# Fonction pour créer des séquences
def create_sequences(X, time_steps=10):
    if len(X) < time_steps:
        X = np.tile(X, (time_steps // len(X) + 1, 1))[:time_steps]
    
    Xs = []
    for i in range(len(X) - time_steps + 1):
        Xs.append(X[i:(i + time_steps)])
    return np.array(Xs)

# Fonction de prétraitement COMPLÈTE
def preprocess_data(df, scaler, encoders, dropped_cols, expected_features):
    """
    Prétraite les données pour qu'elles correspondent exactement au format d'entraînement
    """
    df_processed = df.copy()
    
    # 1. Normaliser les noms de colonnes (tout en minuscules)
    df_processed.columns = df_processed.columns.str.lower()
    
    # 2. Supprimer l'ID si présent
    if 'id' in df_processed.columns:
        df_processed = df_processed.drop('id', axis=1)
    
    # 3. Mapper les noms de colonnes du CSV vers ceux du dataset d'entraînement
    column_mapping = {
        'sload': 'sload',
        'dload': 'dload', 
        'spkts': 'spkts',
        'dpkts': 'dpkts',
        'stime': 'stime',
        'ltime': 'ltime',
        'sintpkt': 'sinpkt',
        'dintpkt': 'dinpkt',
        'smeansz': 'smean',
        'dmeansz': 'dmean'
    }
    df_processed = df_processed.rename(columns=column_mapping)
    
    # 4. Calculer les colonnes dérivées si nécessaire
    if 'rate' not in df_processed.columns or df_processed['rate'].isna().all():
        if 'dur' in df_processed.columns and 'spkts' in df_processed.columns:
            df_processed['rate'] = df_processed.apply(
                lambda row: row['spkts'] / row['dur'] if row['dur'] > 0 else 0,
                axis=1
            )
    
    if 'smean' not in df_processed.columns and 'sbytes' in df_processed.columns and 'spkts' in df_processed.columns:
        df_processed['smean'] = df_processed.apply(
            lambda row: int(row['sbytes'] / row['spkts']) if row['spkts'] > 0 else 0,
            axis=1
        )
    
    if 'dmean' not in df_processed.columns and 'dbytes' in df_processed.columns and 'dpkts' in df_processed.columns:
        df_processed['dmean'] = df_processed.apply(
            lambda row: int(row['dbytes'] / row['dpkts']) if row['dpkts'] > 0 else 0,
            axis=1
        )
    
    # 5. Encoder les variables catégorielles AVANT de supprimer les colonnes
    for col in ['proto', 'service', 'state']:
        if col in df_processed.columns and col in encoders:
            le = encoders[col]
            df_processed[col] = df_processed[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df_processed[col] = le.transform(df_processed[col])
    
    # 6. Supprimer label et attack_cat si présents
    for col in ['label', 'attack_cat']:
        if col in df_processed.columns:
            df_processed = df_processed.drop(col, axis=1)
    
    # 7. Supprimer les colonnes hautement corrélées
    for col in dropped_cols:
        if col in df_processed.columns:
            df_processed = df_processed.drop(col, axis=1)
    
    # 8. Si on a les noms de features attendues, s'assurer qu'on les a toutes
    if expected_features is not None:
        # Ajouter les colonnes manquantes avec des valeurs par défaut
        for col in expected_features:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # Réorganiser les colonnes dans le bon ordre
        df_processed = df_processed[expected_features]
    
    # 9. Normaliser avec le scaler
    X_scaled = scaler.transform(df_processed)
    
    return X_scaled

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/security-checked.png", width=150)
    st.markdown("### 📊 Options")
    
    demo_mode = st.radio(
        "Mode de démonstration:",
        ["📁 Charger un fichier CSV", "✍️ Saisie manuelle", "🎲 Données aléatoires"]
    )
    
    st.markdown("---")
    st.markdown("### 📈 Statistiques du modèle")
    st.metric("Précision", "97%")
    st.metric("Recall", "97%")
    st.metric("F1-Score", "97%")
    
    st.markdown("---")
    
    # Afficher les informations sur les features si disponibles
    if expected_features is not None:
        with st.expander("ℹ️ Features du modèle"):
            st.write(f"**Nombre de features:** {len(expected_features)}")
            st.write(f"**Colonnes supprimées:** {len(dropped_cols)}")
    
    st.markdown("---")
    st.info("Modèle: GRU Bidirectionnel\nDataset: UNSW-NB15\nDéveloppé par: DJILI & El Kadiri")

# Corps principal
if model is None:
    st.error("⚠️ Impossible de charger le modèle. Vérifiez que tous les fichiers sont présents dans le dossier 'models/'.")
elif expected_features is None:
    st.warning("⚠️ Impossible de récupérer les noms des features. Le scaler ne contient pas 'feature_names_in_'.")
else:
    # Mode 1: Upload de fichier CSV
    if demo_mode == "📁 Charger un fichier CSV":
        st.markdown("### 📂 Chargement de données")
        
        uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Fichier chargé avec succès! {len(df)} enregistrements détectés.")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Aperçu des données")
                    st.dataframe(df.head(10), use_container_width=True)
                
                with col2:
                    st.markdown("#### Informations")
                    st.write(f"**Nombre de lignes:** {len(df)}")
                    st.write(f"**Nombre de colonnes:** {len(df.columns)}")
                
                if st.button("🔍 Analyser le trafic", type="primary", use_container_width=True):
                    with st.spinner("Analyse en cours..."):
                        try:
                            # Prétraitement
                            X_processed = preprocess_data(df, scaler, encoders, dropped_cols, expected_features)
                            
                            # Création des séquences
                            X_seq = create_sequences(X_processed, time_steps=10)
                            
                            # Prédictions
                            predictions = model.predict(X_seq, verbose=0)
                            predictions_binary = (predictions > 0.5).astype(int).flatten()
                            
                            # Résultats
                            n_attacks = np.sum(predictions_binary)
                            n_normal = len(predictions_binary) - n_attacks
                            attack_percentage = (n_attacks / len(predictions_binary)) * 100
                            
                            st.markdown("---")
                            st.markdown("### 📊 Résultats de l'analyse")
                            
                            # Métriques
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total analysé", len(predictions_binary))
                            
                            with col2:
                                st.metric("Trafic Normal", n_normal, delta=f"{100-attack_percentage:.1f}%")
                            
                            with col3:
                                st.metric("Attaques détectées", n_attacks, delta=f"{attack_percentage:.1f}%", delta_color="inverse")
                            
                            with col4:
                                if attack_percentage > 50:
                                    st.metric("Niveau de menace", "ÉLEVÉ ⚠️")
                                elif attack_percentage > 20:
                                    st.metric("Niveau de menace", "MOYEN ⚡")
                                else:
                                    st.metric("Niveau de menace", "FAIBLE ✅")
                            
                            # Graphiques
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_pie = go.Figure(data=[go.Pie(
                                    labels=['Normal', 'Attaque'],
                                    values=[n_normal, n_attacks],
                                    hole=.3,
                                    marker_colors=['#00C851', '#ff4444']
                                )])
                                fig_pie.update_layout(title="Distribution du trafic", height=400)
                                st.plotly_chart(fig_pie, use_container_width=True)
                            
                            with col2:
                                fig_hist = go.Figure(data=[go.Histogram(
                                    x=predictions.flatten(),
                                    nbinsx=50,
                                    marker_color='#667eea'
                                )])
                                fig_hist.update_layout(
                                    title="Distribution des probabilités d'attaque",
                                    xaxis_title="Probabilité",
                                    yaxis_title="Fréquence",
                                    height=400
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            # Tableau détaillé
                            st.markdown("### 📋 Détails des prédictions")
                            results_df = pd.DataFrame({
                                'Index': range(len(predictions_binary)),
                                'Probabilité': predictions.flatten(),
                                'Prédiction': ['Attaque' if p == 1 else 'Normal' for p in predictions_binary],
                                'Confiance': [f"{p*100:.1f}%" if predictions_binary[i] == 1 else f"{(1-p)*100:.1f}%" 
                                             for i, p in enumerate(predictions.flatten())]
                            })
                            
                            filter_option = st.selectbox("Filtrer par:", ["Tous", "Attaques uniquement", "Normal uniquement"])
                            
                            if filter_option == "Attaques uniquement":
                                results_df = results_df[results_df['Prédiction'] == 'Attaque']
                            elif filter_option == "Normal uniquement":
                                results_df = results_df[results_df['Prédiction'] == 'Normal']
                            
                            st.dataframe(
                                results_df.style.applymap(
                                    lambda x: 'background-color: #ffcccc' if x == 'Attaque' else 
                                             ('background-color: #ccffcc' if x == 'Normal' else ''),
                                    subset=['Prédiction']
                                ),
                                use_container_width=True,
                                height=400
                            )
                            
                            # Téléchargement
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Télécharger les résultats (CSV)",
                                data=csv,
                                file_name=f"ids_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        except Exception as e:
                            st.error(f"❌ Erreur lors de l'analyse: {e}")
                            with st.expander("🔍 Détails de l'erreur (pour debug)"):
                                import traceback
                                st.code(traceback.format_exc())
                        
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement: {e}")
    
    # Mode 2: Saisie manuelle
    elif demo_mode == "✍️ Saisie manuelle":
        st.markdown("### ✍️ Saisie manuelle des paramètres réseau")
        st.info("Entrez les caractéristiques d'un paquet réseau pour analyse")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            proto = st.selectbox("Protocole", ['tcp', 'udp', 'icmp', 'arp', 'idrp'])
            state = st.selectbox("État", ['FIN', 'INT', 'CON', 'REQ', 'RST'])
            service = st.selectbox("Service", ['-', 'dns', 'http', 'ftp', 'ssh'])
        
        with col2:
            dur = st.number_input("Durée (s)", 0.0, 1000.0, 0.000007, format="%.6f")
            spkts = st.number_input("Paquets Source", 0, 10000, 2)
            dpkts = st.number_input("Paquets Destination", 0, 10000, 0)
        
        with col3:
            sttl = st.number_input("TTL Source", 0, 255, 60)
            dttl = st.number_input("TTL Destination", 0, 255, 0)
            sload = st.number_input("Charge Source", 0.0, 1e9, 150857136.0, format="%.1f")
        
        if st.button("🔍 Analyser ce paquet", type="primary", use_container_width=True):
            # Créer un DataFrame minimal
            manual_data = pd.DataFrame([{
                'proto': proto, 'service': service, 'state': state,
                'dur': dur, 'spkts': spkts, 'dpkts': dpkts,
                'sttl': sttl, 'dttl': dttl, 'sload': sload,
                'dload': 0.0, 'sloss': 0, 'dloss': 0,
                'swin': 0, 'dwin': 0, 'stcpb': 0, 'dtcpb': 0,
                'trans_depth': 0, 'response_body_len': 0,
                'sjit': 0.0, 'djit': 0.0, 
                'sinpkt': 0.007, 'dinpkt': 0.0, 
                'tcprtt': 0.0, 'synack': 0.0, 'ackdat': 0.0,
                'is_sm_ips_ports': 0, 'ct_state_ttl': 0, 
                'ct_flw_http_mthd': 0, 'is_ftp_login': 0, 
                'ct_ftp_cmd': 0, 'ct_srv_src': 29, 'ct_srv_dst': 29,
                'ct_dst_ltm': 9, 'ct_src_ltm': 9, 
                'ct_src_dport_ltm': 9, 'ct_dst_sport_ltm': 9, 
                'ct_dst_src_ltm': 29
            }])
            
            try:
                with st.spinner("Analyse en cours..."):
                    X_processed = preprocess_data(manual_data, scaler, encoders, dropped_cols, expected_features)
                    X_seq = create_sequences(X_processed, time_steps=10)
                    prediction = model.predict(X_seq, verbose=0)[0][0]
                    is_attack = prediction > 0.5
                    
                    st.markdown("---")
                    st.markdown("### 🎯 Résultat de l'analyse")
                    
                    if is_attack:
                        st.markdown(f'<div class="attack-alert">⚠️ ATTAQUE DÉTECTÉE - Confiance: {prediction*100:.2f}%</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="normal-alert">✅ TRAFIC NORMAL - Confiance: {(1-prediction)*100:.2f}%</div>', unsafe_allow_html=True)
                    
                    # Jauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prediction * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probabilité d'attaque (%)"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkred" if is_attack else "darkgreen"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgreen"},
                                {'range': [50, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
                with st.expander("🔍 Détails"):
                    import traceback
                    st.code(traceback.format_exc())
    
    # Mode 3: Données aléatoires
    else:
        st.markdown("### 🎲 Génération de données aléatoires")
        
        n_samples = st.slider("Nombre d'échantillons:", 10, 500, 100)
        
        if st.button("🎲 Générer et analyser", type="primary", use_container_width=True):
            try:
                if os.path.exists('data_sample/unsw_nb15_demo_binary_2000.csv'):
                    sample_df = pd.read_csv('data_sample/unsw_nb15_demo_binary_2000.csv')
                    random_df = sample_df.sample(n=min(n_samples, len(sample_df)))
                else:
                    st.error("Fichier de données d'exemple introuvable!")
                    st.stop()
                
                with st.spinner("Analyse en cours..."):
                    X_processed = preprocess_data(random_df, scaler, encoders, dropped_cols, expected_features)
                    X_seq = create_sequences(X_processed, time_steps=10)
                    predictions = model.predict(X_seq, verbose=0)
                    predictions_binary = (predictions > 0.5).astype(int).flatten()
                    
                    n_attacks = np.sum(predictions_binary)
                    n_normal = len(predictions_binary) - n_attacks
                    attack_percentage = (n_attacks / len(predictions_binary)) * 100
                    
                    st.markdown("---")
                    st.markdown("### 📊 Résultats")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Analysés", len(predictions_binary))
                    with col2:
                        st.metric("Normal", n_normal, delta=f"{100-attack_percentage:.1f}%")
                    with col3:
                        st.metric("Attaques", n_attacks, delta=f"{attack_percentage:.1f}%", delta_color="inverse")
                    
                    # Graphique temporel
                    fig_timeline = go.Figure()
                    fig_timeline.add_trace(go.Scatter(
                        y=predictions.flatten(),
                        mode='lines+markers',
                        name='Probabilité',
                        line=dict(color='#667eea', width=2),
                        marker=dict(
                            size=6,
                            color=predictions_binary,
                            colorscale=[[0, '#00C851'], [1, '#ff4444']],
                            showscale=True
                        )
                    ))
                    fig_timeline.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Seuil")
                    fig_timeline.update_layout(
                        title="Évolution temporelle",
                        xaxis_title="Index",
                        yaxis_title="Probabilité",
                        height=400
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
                    
            except Exception as e:
                st.error(f"❌ Erreur: {e}")
                with st.expander("🔍 Détails"):
                    import traceback
                    st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>🛡️ Système de Détection d'Intrusions - Projet ML/DL pour Cybersécurité</p>
        <p>Développé par DJILI Mohamed Amine & El Kadiri Omar</p>
    </div>
""", unsafe_allow_html=True)
