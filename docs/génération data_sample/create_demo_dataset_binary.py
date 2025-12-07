# create_demo_dataset_binary.py
import pandas as pd
import os
from sklearn.utils import shuffle

# === CONFIGURATION ===
FILES = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv"
]

# Colonnes exactes du dataset UNSW-NB15 (ordre obligatoire !)
COLUMNS = [
    'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss','dloss',
    'service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth',
    'response_body_len','sjit','djit','Stime','Ltime','Sintpkt','Dintpkt','tcprtt','synack','ackdat',
    'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst',
    'ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm',
    'attack_cat','Label'
]

# Nombre d'échantillons par classe dans la démo (500 → fichier de ~5 Mo, charge en <1s dans Streamlit)
SAMPLES_PER_CLASS = 1000  # 1000 Normal + 1000 Attaque = 2000 lignes → parfait équilibre

print("Chargement des 4 fichiers UNSW-NB15...")

dfs = []
for f in FILES:
    if os.path.exists(f):
        print(f"  → Lecture {f}")
        df = pd.read_csv(f, header=None, names=COLUMNS, low_memory=False)
        dfs.append(df)
    else:
        print(f"  Warning: {f} non trouvé !")

if not dfs:
    raise FileNotFoundError("Aucun fichier UNSW-NB15_*.csv trouvé dans le dossier courant !")

data = pd.concat(dfs, ignore_index=True)
print(f"Total chargé : {len(data):,} lignes")

# Nettoyage basique
data = data.dropna(subset=['Label'])
data['Label'] = data['Label'].astype(int)

# === ÉCHANTILLONNAGE ÉQUILIBRÉ BINAIRE ===
normal = data[data['Label'] == 0]
attack = data[data['Label'] == 1]

print(f"Normal  (0) : {len(normal):,} échantillons")
print(f"Attaque (1) : {len(attack):,} échantillons")

# Prendre SAMPLES_PER_CLASS de chaque
normal_sample = normal.sample(n=SAMPLES_PER_CLASS, random_state=42)
attack_sample = attack.sample(n=SAMPLES_PER_CLASS, random_state=42)

# Fusion + mélange
demo_df = pd.concat([normal_sample, attack_sample], ignore_index=True)
demo_df = shuffle(demo_df, random_state=42)

# Suppression des colonnes inutiles pour la démo (optionnel, garde juste les features numériques + Label)
# On garde TOUTES les 47 features numériques + Label (on enlève srcip, dstip, attack_cat si tu veux)
demo_df = demo_df.drop(columns=['srcip', 'dstip', 'attack_cat'], errors='ignore')

# Sauvegarde
os.makedirs("data_sample", exist_ok=True)
output_path = "data_sample/unsw_nb15_demo_binary_2000.csv"
demo_df.to_csv(output_path, index=False)

print("\nFichier de démo binaire créé avec succès !")
print(f"    → {output_path}")
print(f"    → {len(demo_df)} lignes (50% Normal / 50% Attaque)")
print(f"    → Prêt pour Streamlit !")
