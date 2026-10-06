import unittest
import pandas as pd
from trajectoryclusteringanalysis.tca import TCA

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
        self.tca = TCA(data=self.df, index_col='id', alphabet=['D', 'C', 'T', 'S'], states=["diagnostiqué", "en soins", "sous traitement", "inf. contrôlée"],
                       mode='unidimensional')

    def test_compute_substitution_cost_matrix(self):
        matrix = self.tca.compute_substitution_cost_matrix()
        self.assertIsNotNone(matrix)

    def test_compute_distance_matrix(self):
        distance_matrix = self.tca.compute_distance_matrix(self.df, metric='hamming')
        self.assertIsNotNone(distance_matrix)

    def test_hierarchical_clustering(self):
        distance_matrix = self.tca.compute_distance_matrix(self.df,metric='hamming')
        linkage_matrix = self.tca.hierarchical_clustering(distance_matrix)
        self.assertIsNotNone(linkage_matrix)

    def test_assign_clusters(self):
        distance_matrix = self.tca.compute_distance_matrix(self.df,metric='hamming')
        linkage_matrix = self.tca.hierarchical_clustering(distance_matrix)
        clusters = self.tca.assign_clusters(linkage_matrix, num_clusters=2)
        self.assertEqual(len(clusters), len(self.df))

    def test_custom_index_column(self):
        data = self.df.rename(columns={'id': 'person_key'})
        tca = TCA(
            data=data,
            index_col='person_key',
            alphabet=['D', 'C', 'T', 'S'],
            states=['D', 'C', 'T', 'S'],
            mode='unidimensional',
        )

        distance_matrix = tca.compute_distance_matrix(data, metric='hamming')
        clusters, _, _ = tca.kmeans_on_wide_format(
            num_clusters=2,
            random_state=0,
            n_init=10,
        )

        self.assertEqual(tca.sequences.shape[0], len(data))
        self.assertEqual(distance_matrix.shape, (len(data), len(data)))
        self.assertEqual(len(clusters), len(data))

    def test_levenshtein_distance_uses_whole_states(self):
        distance_matrix = self.tca.compute_distance_matrix(self.df, metric='levenshtein')

        self.assertEqual(distance_matrix[0, 1], 2)
        self.assertEqual(distance_matrix[1, 0], 2)

if __name__ == '__main__':
    unittest.main()