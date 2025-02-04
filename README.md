# Trajectory Clustering Analysis (TCA)

## Description

Trajectory Clustering Analysis (TCA) is a Python package for analyzing and visualizing temporal treatment sequences in a dataset.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/ndiaga21/TrajectoryClusteringAnalysis.git


   # Trajectory Clustering Analysis (TCA)

## 🚀 Introduction

**Trajectory Clustering Analysis (TCA)** est un package Python conçu pour l'analyse des trajectoires de soins à l'aide de techniques de clustering. Il permet de modéliser, regrouper et visualiser des séquences de traitements médicaux afin d'identifier des patterns et des profils de patients similaires.

## 🔍 Fonctionnalités principales

- **Modélisation des trajectoires de soins** : Représentation des patients par des séquences chronologiques de traitements.
- **Clustering des trajectoires** : Utilisation de mesures de dissimilarité (comme la distance de Hamming, OM,dtw) combinées à des méthodes de clustering hiérarchique (CAH).
- **Visualisation des trajectoires** : Représentation graphique des trajectoires pour une meilleure interprétation des résultats.
- **Gestion des logs** : Suivi des exécutions grâce au module `logger.py`.

## 📦 Installation

1. Clonez le dépôt :
   ```bash
   git clone <lien_du_repo>
   cd TrajectoryClusteringAnalysis
   ```

2. Créez un environnement virtuel (optionnel mais recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

3. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

4. Installez le package :
   ```bash
   pip install .
   ```

## ⚙️ Utilisation de base

```python
from TrajectoryClusteringAnalysis import TCA

# Exemple de données
trajectories = [
    ["Chirurgie", "Chimiothérapie", "Radiothérapie"],
    ["Chimiothérapie", "Radiothérapie"],
    ["Chirurgie", "Radiothérapie"]
]
## Preporocessing data

# Initialisation et clustering
model = TCA.TCA(data=df,
              id='id',
              alphabet=["Chirurgie", "Chimiothérapie", "Radiothérapie"],
              states=["Chirurgie", "Chimiothérapie", "Radiothérapie"])
#compute distance
model.compute_distance_matrix(metric='hamming', substitution_cost_matrix=None)
## Clustering CAH:
linkage_matrix = model.hierarchical_clustering(distance_matrix)
model.plot_dendrogram(linkage_matrix)
# Visualisation
model.plot_clustermap(linkage_matrix)
#assigne clusters
clusters = tca.assign_clusters(linkage_matrix, num_clusters=4)
model.plot_cluster_heatmaps(clusters)
```

## 📊 Structure du projet

```
TrajectoryClusteringAnalysis/
├── data/                   # Données d'exemple ou de test
├── Notebook/               # Notebooks d'analyse et de démonstration
├── TrajectoryClusteringAnalysis/
│   ├── TCA.py              # Méthodes de clustering des trajectoires
│   └── logger.py           # Module de gestion des logs
├── venv/                   # Environnement virtuel
├── setup.py                # Script d'installation
├── requirements.txt        # Dépendances
└── README.md               # Documentation
```

## 🧪 Exemples

Des notebooks d'exemple sont disponibles dans le dossier `Notebook` pour illustrer différentes analyses de trajectoires.

## 🤝 Contribuer

1. Fork le projet
2. Créez votre branche de fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Commitez vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Poussez la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

## 📧 Contact

**Auteur :** Nicolas & Ndiaga  
**Email :** ndiagadiengs1@gmail.com

---

© 2024 - Trajectory Clustering Analysis (TCA). Tous droits réservés.


