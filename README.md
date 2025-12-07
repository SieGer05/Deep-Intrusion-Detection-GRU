<h1 align="center">Système de Détection d'Intrusion Réseau (IDS)</h1>
<h3 align="center">Deep Learning (Bi-GRU) & Application de Surveillance Temps Réel</h3>

<p align="center">
  <img src="./docs/The-unfold-form-of-a-Bidirectional-GRU.ppm" alt="GRU Architecture" width="600">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.10%2B-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B" alt="Streamlit">
  <img src="https://img.shields.io/badge/Dataset-UNSW--NB15-green" alt="Dataset">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen" alt="Status">
</p>

<p align="center">
  <strong>Auteurs :</strong> DJILI Mohamed Amine & El Kadiri Omar
</p>

<hr>

<h2>1. Présentation du Projet</h2>

<p align="justify">
Ce projet implémente un <strong>Système de Détection d'Intrusion (IDS)</strong> complet, allant de l'entraînement d'un modèle de Deep Learning avancé jusqu'à son déploiement dans une interface web de démonstration.
</p>
<p align="justify">
Le système est capable de classifier le trafic réseau en <strong>Normal</strong> ou <strong>Attaque</strong> avec une précision d'environ <strong>98%</strong> sur le dataset complexe <strong>UNSW-NB15</strong>. Il utilise une architecture de réseaux de neurones récurrents (GRU Bidirectionnel) pour analyser les séquences temporelles du trafic.
</p>

<h2>2. Architecture Technique</h2>

<h3>Pipeline de Données (Data Pipeline)</h3>
Pour garantir la fiabilité entre l'entraînement et l'inférence (l'application), le pipeline (défini dans <code>utils.py</code>) suit ces étapes :
<ul>
  <li><strong>Nettoyage :</strong> Suppression des features fortement corrélées via une liste d'exclusion persistante (<code>dropped_columns.pkl</code>).</li>
  <li><strong>Encodage :</strong> Transformation des variables catégorielles (proto, service, state) via des <code>LabelEncoders</code> sauvegardés.</li>
  <li><strong>Normalisation :</strong> Application d'un <code>StandardScaler</code> pour centrer-réduire les données numériques.</li>
  <li><strong>Séquençage :</strong> Transformation des données en fenêtres temporelles (Time Steps = 10) pour l'analyse contextuelle par le RNN.</li>
</ul>

<h3>Modèle Deep Learning</h3>
L'architecture du modèle (API Keras/TensorFlow) est conçue pour la robustesse :
<ol>
  <li><strong>Input Layer :</strong> Séquences temporelles (10 pas de temps).</li>
  <li><strong>Bidirectional GRU :</strong> 2 couches (64 et 32 unités) pour capturer le contexte passé et futur.</li>
  <li><strong>Régularisation :</strong> Dropout (0.5) et pénalité L2 pour éviter le surapprentissage.</li>
  <li><strong>Batch Normalization :</strong> Pour accélérer et stabiliser la convergence.</li>
  <li><strong>Output :</strong> Neurone Sigmoid pour la classification binaire.</li>
</ol>

<h2>3. Structure du Projet</h2>

<pre>
.
├── app.py                      # Application Web de démonstration (Interface Streamlit)
├── requirements.txt            # Liste des dépendances Python
├── unsw_nb15_RNN_LSTM.ipynb    # Notebook Jupyter d'entraînement et d'analyse (EDA)
│
├── models/                     # Artefacts du modèle (Sauvegardés après entraînement)
│   ├── ids_gru_model.keras     # Le modèle de Deep Learning compilé
│   ├── scaler_std.pkl          # Le scaler (StandardScaler)
│   ├── label_encoders.pkl      # Dictionnaire des encodeurs catégoriels
│   └── dropped_columns.pkl     # Liste des colonnes ignorées
│
├── data_sample/                # Données pour la démonstration
│   └── unsw_nb15_demo_binary_2000.csv  # Échantillon de 2000 lignes réelles pour test
│
├── UNSW-NB15-Dataset/          # Dataset original complet (non tracké par git)
├── docs/                       # Documentation et images d'architecture
└── .gitignore                  # Configuration Git
</pre>

<h2>4. Fonctionnalités de l'Application</h2>

L'interface utilisateur (<code>app.py</code>) offre trois modes d'interaction :
1.  <strong>Test Unitaire :</strong> Sélectionne un paquet réseau aléatoire, le traite et affiche la probabilité d'attaque en temps réel.
2.  <strong>Évaluation Complète :</strong> Lance un test sur 2000 échantillons pour vérifier la précision globale du modèle en direct.
3.  <strong>Statistiques :</strong> Visualisation de la distribution des données de test.

<h2>5. Installation et Lancement</h2>

<h3>Pré-requis</h3>
<ul>
    <li>Python 3.8 ou supérieur</li>
</ul>

<h3>Installation des dépendances</h3>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>Lancer l'Application Web (Démo)</h3>
Pour utiliser l'interface graphique de détection :
<pre><code>streamlit run app.py</code></pre>
<p>L'application s'ouvrira automatiquement dans votre navigateur à l'adresse <code>http://localhost:8501</code>.</p>

<h3>Ré-entraîner le modèle</h3>
Si vous souhaitez reproduire l'entraînement ou modifier l'architecture :
<pre><code>jupyter notebook unsw_nb15_RNN_LSTM.ipynb</code></pre>

<h2>6. Performances</h2>

Les résultats obtenus sur le jeu de test (82k échantillons) :

<div align="center">
  <table border="1">
    <thead>
      <tr>
        <th>Métrique</th>
        <th>Résultat Global</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><strong>Accuracy</strong></td>
        <td><strong>~ 97-98%</strong></td>
      </tr>
      <tr>
        <td><strong>F1-Score (Attaque)</strong></td>
        <td>0.98</td>
      </tr>
       <tr>
        <td><strong>F1-Score (Normal)</strong></td>
        <td>0.97</td>
      </tr>
    </tbody>
  </table>
</div>

<h2>7. Auteurs</h2>

<p>Projet réalisé dans le cadre du cursus d'ingénierie en Cybersécurité.</p>
<ul>
  <li><strong>DJILI Mohamed Amine</strong></li>
  <li><strong>El Kadiri Omar</strong></li>
</ul>
