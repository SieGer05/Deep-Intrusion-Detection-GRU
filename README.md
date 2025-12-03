<h1 align="center">Système de Détection d'Intrusion Réseau (IDS)</h1>
<h3 align="center">Basé sur l'architecture Deep Learning GRU Bidirectionnel</h3>

<p align="center">
   <img src="./docs/The-unfold-form-of-a-Bidirectional-GRU.ppm" alt="GRU Architecture">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.0%2B-orange" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Dataset-UNSW--NB15-green" alt="Dataset">
</p>

<p align="center">
  <strong>Auteurs :</strong> DJILI Mohamed Amine & El Kadiri Omar
</p>

<hr>

<h2>1. Présentation du Projet</h2>

<p align="justify">
Ce projet implémente un Système de Détection d'Intrusion (IDS) de nouvelle génération utilisant des techniques avancées de Deep Learning. L'objectif est de classifier le trafic réseau comme <strong>Normal</strong> ou <strong>Attaque</strong> avec une haute précision et un taux de faux positifs minimal.
</p>

<p align="justify">
Le modèle est entraîné sur le dataset <strong>UNSW-NB15</strong>, reconnu pour sa complexité et sa représentation réaliste des menaces modernes. L'architecture repose sur des réseaux de neurones récurrents de type <strong>GRU (Gated Recurrent Unit) Bidirectionnel</strong>, optimisés pour traiter les données séquentielles temporelles.
</p>

<h2>2. Architecture Technique</h2>

<h3>Méthodologie de Pré-traitement</h3>
<ul>
  <li><strong>Nettoyage de Données :</strong> Suppression des features hautement corrélées (> 95%) pour réduire le bruit.</li>
  <li><strong>Standardisation :</strong> Utilisation de <code>StandardScaler</code> pour normaliser les données (Moyenne = 0, Écart-type = 1), assurant une convergence rapide du modèle.</li>
  <li><strong>Gestion du Déséquilibre :</strong> Calcul et application de <code>class_weights</code> pour équilibrer l'apprentissage entre les classes majoritaires et minoritaires.</li>
  <li><strong>Séquençage :</strong> Transformation des données en fenêtres temporelles glissantes (Time Steps) pour l'analyse contextuelle.</li>
</ul>

<h3>Modèle Deep Learning</h3>
Le modèle utilise l'API Keras (TensorFlow) avec la structure suivante :
<ol>
  <li><strong>Input Layer :</strong> Séquences temporelles.</li>
  <li><strong>Bidirectional GRU :</strong> Capture du contexte passé et futur du flux réseau.</li>
  <li><strong>Régularisation :</strong> Application de <code>Dropout (0.5)</code> et de régularisation <code>L2</code> pour empêcher le surapprentissage (Overfitting).</li>
  <li><strong>Batch Normalization :</strong> Stabilisation de l'apprentissage.</li>
  <li><strong>Output Layer :</strong> Sigmoid pour la classification binaire.</li>
</ol>

<h2>3. Structure du Répertoire</h2>

<pre>
.
├── models/                  # Contient le modèle entraîné et le scaler
│   ├── ids_gru_model.keras  # Le fichier modèle Deep Learning
│   └── scaler_std.pkl       # Le fichier de normalisation (Joblib)
├── requirements.txt         # Liste des dépendances Python
├── .gitignore               # Configuration pour ignorer le dataset et les fichiers temporaires
├── unsw_nb15_RNN_LSTM.ipynb # Notebook principal (Entraînement et Évaluation)
└── README.md                # Documentation du projet
</pre>

<h2>4. Performances</h2>

Les résultats obtenus sur le jeu de test démontrent la robustesse de l'approche :

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
        <td><strong>Accuracy (Précision Globale)</strong></td>
        <td><strong>~ 98%</strong></td>
      </tr>
      <tr>
        <td><strong>Perte (Loss)</strong></td>
        <td>< 0.10</td>
      </tr>
    </tbody>
  </table>
</div>

<br>
<div align="center">
  <table border="1">
    <thead>
      <tr>
        <th>Classe</th>
        <th>Précision</th>
        <th>Rappel (Recall)</th>
        <th>F1-Score</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Normal</td>
        <td>0.98</td>
        <td>0.97</td>
        <td>0.97</td>
      </tr>
      <tr>
        <td>Attaque</td>
        <td>0.97</td>
        <td>0.98</td>
        <td>0.98</td>
      </tr>
    </tbody>
  </table>
</div>

<h2>5. Installation et Utilisation</h2>

<h3>Pré-requis</h3>
<ul>
    <li>Python 3.8 ou supérieur</li>
    <li>Un environnement virtuel est recommandé</li>
</ul>

<h3>Installation</h3>
<pre><code>pip install -r requirements.txt</code></pre>

<h3>Lancement</h3>
<p>
Le projet est contenu dans un Jupyter Notebook. Vous pouvez l'exécuter via Jupyter Lab, Jupyter Notebook ou Google Colab.
</p>

<pre><code>jupyter notebook unsw_nb15_RNN_LSTM.ipynb</code></pre>

<h2>6. Auteurs</h2>

<p>Ce projet a été réalisé dans le cadre d'études en ingénierie de Cybersécurité.</p>
<ul>
  <li><strong>DJILI Mohamed Amine</strong></li>
  <li><strong>El Kadiri Omar</strong></li>
</ul>