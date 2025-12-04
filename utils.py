# utils.py → Version 100% alignée avec l'entraînement

import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# =============================================
# Chargement du modèle et scaler
# =============================================
model = load_model("models/ids_gru_model.keras")
with open("models/scaler_std.pkl", "rb") as f:
    scaler = pickle.load(f)

# =============================================
# IMPORTANT : Recréer les LabelEncoders EXACTEMENT comme à l'entraînement
# =============================================
# Vous devez sauvegarder les encoders pendant l'entraînement avec :
# joblib.dump({'proto': le_proto, 'service': le_service, ...}, 'models/label_encoders.pkl')

try:
    import joblib
    encoders = joblib.load("models/label_encoders.pkl")
    print("✅ LabelEncoders chargés depuis le fichier")
except FileNotFoundError:
    print("⚠️ ATTENTION : label_encoders.pkl introuvable !")
    print("   Les prédictions peuvent être incorrectes.")
    print("   Veuillez sauvegarder les encoders lors de l'entraînement.")
    encoders = None

# =============================================
# Liste des colonnes supprimées (corrélation > 0.95)
# =============================================
# VOUS DEVEZ METTRE ICI les colonnes exactes supprimées lors de l'entraînement
# Exemple basé sur UNSW-NB15 typique :
DROPPED_CORR_COLUMNS = [
    # 'sttl',  # souvent corrélé avec 'dttl'
    # 'dload', # souvent corrélé avec 'dbytes'
    # 'ct_srv_dst', # etc.
]

# =============================================
# Encodage des colonnes catégorielles
# =============================================
def encode_categorical(df):
    """
    Encode les colonnes catégorielles EXACTEMENT comme à l'entraînement.
    """
    df = df.copy()
    
    cols_to_encode = ['proto', 'service', 'state', 'attack_cat']
    
    if encoders is not None:
        # Utiliser les encoders sauvegardés
        for col in cols_to_encode:
            if col in df.columns:
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except KeyError:
                    # Valeur inconnue → assigner -1
                    df[col] = df[col].apply(
                        lambda x: encoders[col].transform([str(x)])[0] 
                        if str(x) in encoders[col].classes_ else -1
                    )
    else:
        # Fallback : encoder manuellement (moins fiable)
        for col in cols_to_encode:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    
    return df

# =============================================
# Prétraitement complet
# =============================================
def preprocess_sample(sample_series):
    """
    Applique TOUS les prétraitements de l'entraînement :
    1. Suppression de 'id' si présent
    2. Encodage des colonnes catégorielles
    3. Suppression des colonnes corrélées
    4. Normalisation
    5. Création de séquence temporelle
    """
    df = sample_series.to_frame().T
    
    # 1. Supprimer 'id' si présent
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # 2. Encoder les colonnes catégorielles
    df = encode_categorical(df)
    
    # 3. Supprimer 'attack_cat' (non utilisé pour la prédiction binaire)
    if 'attack_cat' in df.columns:
        df = df.drop('attack_cat', axis=1)
    
    # 4. Supprimer les colonnes corrélées (si elles existent)
    for col in DROPPED_CORR_COLUMNS:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # 5. Convertir en array
    X = df.values.astype(np.float32)
    
    # 6. Normalisation
    X_scaled = scaler.transform(X)
    
    return X_scaled

# =============================================
# Création de séquence temporelle
# =============================================
def create_single_sequence(X_scaled, time_steps=10):
    """
    Crée une séquence temporelle pour une seule prédiction.
    Comme vous utilisez TIME_STEPS=10, on répète l'échantillon.
    """
    # Répéter l'échantillon pour créer une "séquence"
    # (alternative : utiliser les 10 derniers échantillons réels si disponibles)
    X_seq = np.repeat(X_scaled, time_steps, axis=0)
    X_seq = X_seq.reshape(1, time_steps, -1)  # (1, 10, features)
    return X_seq

# =============================================
# Fonction de prédiction
# =============================================
def predict_sample(features_or_series, time_steps=10):
    """
    Prédit si un échantillon est une attaque ou du trafic normal.
    
    Args:
        features_or_series: pd.Series (ligne du DataFrame)
        time_steps: nombre de timesteps (doit être 10 comme à l'entraînement)
    
    Returns:
        prediction (str): "Normal" ou "Attaque"
        confidence (float): Confiance de la prédiction
        proba_attack (float): Probabilité d'attaque [0, 1]
    """
    # Prétraitement complet
    X_scaled = preprocess_sample(features_or_series)
    
    # Création de la séquence temporelle
    X_seq = create_single_sequence(X_scaled, time_steps)
    
    # Prédiction
    proba_attack = float(model.predict(X_seq, verbose=0)[0][0])
    
    # Classification
    if proba_attack > 0.5:
        prediction = "Attaque"
        confidence = proba_attack
    else:
        prediction = "Normal"
        confidence = 1 - proba_attack
    
    return prediction, confidence, proba_attack
