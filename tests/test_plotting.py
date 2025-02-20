import unittest
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster,leaves_list
from TrajectoryClusteringAnalysis.plotting import plot_dendrogram, plot_clustermap, plot_inertia, plot_cluster_heatmaps, plot_treatment_percentage, bar_treatment_percentage

class TestPlotting(unittest.TestCase):

    def setUp(self):
        # Générer un DataFrame avec 10 individus et 6 mois de suivi
        data = {
            'id': list(range(1, 11)),
            'month_1': np.random.choice(['D', 'C', 'T', 'S'], 10),
            'month_2': np.random.choice(['D', 'C', 'T', 'S'], 10),
            'month_3': np.random.choice(['D', 'C', 'T', 'S'], 10),
            'month_4': np.random.choice(['D', 'C', 'T', 'S'], 10),
            'month_5': np.random.choice(['D', 'C', 'T', 'S'], 10),
            'month_6': np.random.choice(['D', 'C', 'T', 'S'], 10)
        }
        self.df = pd.DataFrame(data)

        X = np.random.rand(10, 2)  
        self.linkage_matrix = linkage(X, method="ward")  # Calculer la matrice de linkag
        # Extraire les clusters (ici on choisit 3 clusters)
        self.clusters = fcluster(self.linkage_matrix, 4, criterion='maxclust')
        self.leaves_order = leaves_list(self.linkage_matrix)
        self.label_to_encoded = {'D': 1, 'C': 2, 'T': 3, 'S': 4}
        self.alphabet = ['D', 'C', 'T', 'S']
        self.states = ["diagnostiqué", "en soins", "sous traitement", "inf. contrôlée"]
        self.colors = 'viridis'

    def test_plot_dendrogram(self):
        plot_dendrogram(self.linkage_matrix)

    def test_plot_clustermap(self):
        plot_clustermap(self.df, 'id', self.label_to_encoded, self.colors, self.alphabet, 
                        self.states , self.linkage_matrix)

    def test_plot_inertia(self):
        plot_inertia(self.linkage_matrix)

    def test_plot_cluster_heatmaps(self):
        plot_cluster_heatmaps(self.df, 'id', self.label_to_encoded, self.colors, 
                              self.alphabet, self.states, self.clusters, self.leaves_order)

    def test_plot_treatment_percentage_no_cluters(self):
        plot_treatment_percentage(self.df, 'id', self.alphabet,self.states)
    
    def test_plot_treatment_percentage_with_cluters(self):
        plot_treatment_percentage(self.df, 'id', self.alphabet,self.states,self.clusters)
    
    def test_bar_treatment_percentage_no_clusters(self):
        bar_treatment_percentage(self.df, 'id', self.alphabet,self.states)
    
    def test_bar_treatment_percentage_with_clusters(self):
        bar_treatment_percentage(self.df, 'id', self.alphabet,self.states, self.clusters)

if __name__ == '__main__':
    unittest.main()