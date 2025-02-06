# Trajectory Clustering Analysis (TCA)

## Description

Trajectory Clustering Analysis (TCA) is a Python package for analyzing and visualizing temporal treatment sequences in a dataset.

## 🚀 Introduction

**Trajectory Clustering Analysis (TCA)** is a Python package designed for the analysis of care trajectories using clustering techniques. It allows modeling, grouping, and visualizing medical treatment sequences to identify patterns and similar patient profiles.

## 🔍 Main Features

- **Modeling Care Trajectories:** Representation of patients through chronological sequences of treatments.
- **Trajectory Clustering:** Utilization of dissimilarity measures (such as Hamming distance, OM, DTW) combined with hierarchical clustering methods (CAH).
- **Trajectory Visualization:** Graphical representation of trajectories for better interpretation of results.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone <repository_link>
   cd TrajectoryClusteringAnalysis
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package:
   ```bash
   pip install .
   ```

## ⚙️ Basic Usage

```python
from TrajectoryClusteringAnalysis.TCA import TCA

# Example data
trajectories = [
    ["Surgery", "Chemotherapy", "Radiotherapy"],
    ["Chemotherapy", "Radiotherapy"],
    ["Surgery", "Radiotherapy"]
]

# Preprocessing data
# (Assuming an image named format_data.png exists in the specified path)
![data_format](TrajectoryClusteringAnalysis/image/format_data.png)

# Initialization and clustering
model = TCA(data=df,
            id='id',
            alphabet=["Surgery", "Chemotherapy", "Radiotherapy"],
            states=["Surgery", "Chemotherapy", "Radiotherapy"])

# Compute distance
model.compute_distance_matrix(metric='hamming', substitution_cost_matrix=None)

# Hierarchical Clustering (CAH)
linkage_matrix = model.hierarchical_clustering(distance_matrix)
model.plot_dendrogram(linkage_matrix)

# Visualization
model.plot_clustermap(linkage_matrix)

# Assign clusters
clusters = model.assign_clusters(linkage_matrix, num_clusters=4)
model.plot_cluster_heatmaps(clusters)
```

## 📊 Project Structure

```
TrajectoryClusteringAnalysis/
├── data/                   # Example or test data
├── Notebook/               # Analysis and demonstration notebooks
├── TrajectoryClusteringAnalysis/
│   ├── __init__.py         # Package initialization
│   ├── TCA.py              # Trajectory clustering methods
│   └── logger.py           # Log management module
├── venv/                   # Virtual environment
├── setup.py                # Installation script
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

## 🧪 Examples

Example notebooks are available in the `Notebook` folder to illustrate different trajectory analyses.

## 🤝 Contributing

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

**Authors:** GREVET Nicolas & DIENG Ndiaga  
**Email:** nicolas.GREVET@univ-amu.fr  
**Email:** ndiaga.diengs1@univ-amu.fr

---

© 2024 - Trajectory Clustering Analysis (TCA). All rights reserved.
