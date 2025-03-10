import pandas as pd
import numpy as np
import logging
from TrajectoryClusteringAnalysis.clustering import compute_substitution_cost_matrix, compute_distance_matrix, hierarchical_clustering, assign_clusters
from TrajectoryClusteringAnalysis.plotting import plot_dendrogram, plot_clustermap, plot_inertia, plot_cluster_heatmaps, plot_treatment_percentage, bar_treatment_percentage

class TCA:
    def __init__(self, data, id, alphabet, states, colors='viridis'):
        self.data = data
        self.id = id
        self.alphabet = alphabet
        self.states = states
        self.colors = colors
        self.leaf_order = None
        self.substitution_cost_matrix = None
        logging.basicConfig(level=logging.INFO)
        
        assert(isinstance(data, pd.DataFrame)), "data must be a pandas DataFrame"
        assert(data.shape[1] > 1), "data must have more than one column"
        assert(data.id.duplicated().sum() == 0), "There are duplicates in the data. Yout dataset must be in long and tidy format "

        print("Dataset :")
        print("data shape: ", self.data.shape)
        mapping_df = pd.DataFrame({'alphabet':self.alphabet, 'label':self.states, 'label encoded':range(1,1+len(self.alphabet))})
        print("state coding:\n", mapping_df)

        data_ready_for_TCA = self.data.copy()
        data_ready_for_TCA['Sequence'] = data_ready_for_TCA.drop(self.id, axis=1).apply(lambda x: '-'.join(x.astype(str)), axis=1)
        data_ready_for_TCA = data_ready_for_TCA[['id', 'Sequence']]
        self.sequences = data_ready_for_TCA['Sequence'].apply(lambda x: np.array([k for k in x.split('-') if k != 'nan'])).to_numpy()

        self.label_to_encoded = mapping_df.set_index('alphabet')['label encoded'].to_dict()
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.info("TCA object initialized successfully")

    def compute_substitution_cost_matrix(self, method='constant', custom_costs=None):
        return compute_substitution_cost_matrix(self.sequences, self.alphabet, method, custom_costs)

    #def optimal_matching(self, seq1, seq2, substitution_cost_matrix, indel_cost=None):
     #   return optimal_matching(seq1, seq2, substitution_cost_matrix, indel_cost, self.alphabet)

    def compute_distance_matrix(self, metric='hamming', substitution_cost_matrix=None):
        return compute_distance_matrix(self.data, self.sequences, self.label_to_encoded, metric, substitution_cost_matrix,self.alphabet)

    def hierarchical_clustering(self, distance_matrix, method='ward', optimal_ordering=True):
        return hierarchical_clustering(self,distance_matrix, method, optimal_ordering)

    def assign_clusters(self, linkage_matrix, num_clusters):
        return assign_clusters(linkage_matrix, num_clusters)

    def plot_dendrogram(self, linkage_matrix):
        plot_dendrogram(linkage_matrix)

    def plot_clustermap(self, linkage_matrix):
        plot_clustermap(self.data, self.id, self.label_to_encoded, self.colors, self.alphabet, self.states, linkage_matrix)

    def plot_inertia(self, linkage_matrix):
        plot_inertia(linkage_matrix)

    def plot_cluster_heatmaps(self, clusters, sorted=True):
        plot_cluster_heatmaps(self.data, self.id, self.label_to_encoded, self.colors, self.alphabet, self.states, clusters, self.leaf_order, sorted)

    def plot_treatment_percentage(self, clusters=None):
        plot_treatment_percentage(self.data, self.id, self.alphabet, self.states, clusters)

    def bar_treatment_percentage(self, clusters=None):
        bar_treatment_percentage(self.data, self.id, self.alphabet, self.states, clusters)

####################################### MAIN #######################################
####################################### MAIN #######################################
####################################### MAIN #######################################
####################################### MAIN #######################################

def main():
    df = pd.read_csv('data/dataframe_test.csv')

    # Sélectionner les colonnes pertinentes pour l'analyse
    selected_cols = df[['id', 'month', 'care_status']]

    # Créer un tableau croisé des données en format large
    #       -> Chaque individu est sur une ligne.
    #       -> Les mesures dans le temps (Temps1, Temps2, Temps3) sont des colonnes distinctes.
    pivoted_data = selected_cols.pivot(index='id', columns='month', values='care_status')
    pivoted_data['id'] = pivoted_data.index
    pivoted_data = pivoted_data[['id'] + [col for col in pivoted_data.columns if col != 'id']]

    # Renommer les colonnes avec un préfixe "month_"
    pivoted_data.columns = ['id'] + ['month_' + str(int(col)+1) for col in pivoted_data.columns[1:]]
    # print(pivoted_data.columns)

    # Sélectionner un échantillon aléatoire de 10% des données
    pivoted_data_random_sample = pivoted_data.sample(frac=0.1, random_state=42).reset_index(drop=True)

    # Filter individuals observed for at least 18 months
    valid_18months_individuals = pivoted_data.dropna(thresh=19).reset_index(drop=True)

    # Select only the first 18 months for analysis
    valid_18months_individuals = valid_18months_individuals[['id'] + [f'month_{i}' for i in range(1, 19)]]

    # print(pivoted_data_random_sample.head())
    # print(data_ready_for_TCA.duplicated().sum())

    # tca = TCA(df_numeriques,state_mapping,colors)
    tca = TCA(data=pivoted_data_random_sample,
              id='id',
              alphabet=['D', 'C', 'T', 'S'],
              states=["diagnostiqué", "en soins", "sous traitement", "inf. contrôlée"])
   
    # tca.plot_treatment_percentages(df_numeriques)

    custom_costs = {'D:C': 1, 'D:T': 2, 'D:S': 3, 'C:T': 1, 'C:S': 2, 'T:S': 1}
    #costs = tca.compute_substitution_cost_matrix(method='custom', custom_costs=custom_costs)
    distance_matrix = tca.compute_distance_matrix(metric='hamming', substitution_cost_matrix=None)
    print("distance matrix :\n",distance_matrix)
    linkage_matrix = tca.hierarchical_clustering(distance_matrix)
    tca.plot_dendrogram(linkage_matrix)
    tca.plot_clustermap(linkage_matrix)
    tca.plot_inertia(linkage_matrix)
    clusters = tca.assign_clusters(linkage_matrix, num_clusters=4)  
    tca.plot_cluster_heatmaps(clusters, sorted=False)
    tca.plot_treatment_percentage()
    tca.plot_treatment_percentage(clusters)
    tca.bar_treatment_percentage()
    tca.bar_treatment_percentage(clusters)
if __name__ == "__main__":
    main()
