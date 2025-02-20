import unittest
import pandas as pd
from TrajectoryClusteringAnalysis.tca import TCA

class TestTCA(unittest.TestCase):

    def setUp(self):
        # Créer un DataFrame de test
        data = {
            'id': [1, 2, 3],
            'month_1': ['D', 'C', 'T'],
            'month_2': ['C', 'T', 'S'],
            'month_3': ['T', 'S', 'D']
        }
        self.df = pd.DataFrame(data)
        self.tca = TCA(data=self.df, id='id', alphabet=['D', 'C', 'T', 'S'], states=["diagnostiqué", "en soins", "sous traitement", "inf. contrôlée"])

    def test_compute_substitution_cost_matrix(self):
        matrix = self.tca.compute_substitution_cost_matrix()
        self.assertIsNotNone(matrix)

    def test_compute_distance_matrix(self):
        distance_matrix = self.tca.compute_distance_matrix(metric='hamming')
        self.assertIsNotNone(distance_matrix)

    def test_hierarchical_clustering(self):
        distance_matrix = self.tca.compute_distance_matrix(metric='hamming')
        linkage_matrix = self.tca.hierarchical_clustering(distance_matrix)
        self.assertIsNotNone(linkage_matrix)

    def test_assign_clusters(self):
        distance_matrix = self.tca.compute_distance_matrix(metric='hamming')
        linkage_matrix = self.tca.hierarchical_clustering(distance_matrix)
        clusters = self.tca.assign_clusters(linkage_matrix, num_clusters=2)
        self.assertEqual(len(clusters), len(self.df))

if __name__ == '__main__':
    unittest.main()