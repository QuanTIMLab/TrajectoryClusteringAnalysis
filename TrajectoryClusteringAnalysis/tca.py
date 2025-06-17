import pandas as pd
import numpy as np
import logging
from TrajectoryClusteringAnalysis.clustering import (
    compute_substitution_cost_matrix,
    compute_distance_matrix,
    hierarchical_clustering,
    assign_clusters
)
from TrajectoryClusteringAnalysis.plotting import (
    plot_dendrogram,
    plot_clustermap,
    plot_inertia,
    plot_cluster_heatmaps,
    plot_treatment_percentage,
    bar_treatment_percentage
)

class TCA:
    """
    Trajectory Clustering Analysis (TCA) class for analyzing and clustering trajectory data.

    Attributes:
        data (pd.DataFrame): Input data in long and tidy format.
        id (str): Column name representing unique identifiers for individuals.
        alphabet (list): List of possible states in the trajectories.
        states (list): Descriptive labels for the states.
        colors (str): Colormap for visualizations (default: 'viridis').
        leaf_order (list): Order of leaves in the dendrogram (default: None).
        substitution_cost_matrix (np.ndarray): Substitution cost matrix (default: None).
    """
    def __init__(self, data, id, alphabet, states, colors='viridis'):
        """
        Initialize the TCA object with input data and parameters.

        Args:
            data (pd.DataFrame): Input data in long and tidy format.
            id (str): Column name representing unique identifiers for individuals.
            alphabet (list): List of possible states in the trajectories.
            states (list): Descriptive labels for the states.
            colors (str): Colormap for visualizations (default: 'viridis').
        """
        self.data = data
        self.id = id
        self.alphabet = alphabet
        self.states = states
        self.colors = colors
        self.leaf_order = None
        self.substitution_cost_matrix = None
        logging.basicConfig(level=logging.INFO)

        # Validate input data
        assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
        assert data.shape[1] > 1, "data must have more than one column"
        assert data.id.duplicated().sum() == 0, "There are duplicates in the data. Your dataset must be in long and tidy format"

        # Print dataset information
        print("Dataset :")
        print("data shape: ", self.data.shape)
        mapping_df = pd.DataFrame({'alphabet': self.alphabet, 'label': self.states, 'label encoded': range(1, 1 + len(self.alphabet))})
        print("state coding:\n", mapping_df)

        # Prepare data for TCA
        data_ready_for_TCA = self.data.copy()
        data_ready_for_TCA['Sequence'] = data_ready_for_TCA.drop(self.id, axis=1).apply(lambda x: '-'.join(x.astype(str)), axis=1)
        data_ready_for_TCA = data_ready_for_TCA[['id', 'Sequence']]
        self.sequences = data_ready_for_TCA['Sequence'].apply(lambda x: np.array([k for k in x.split('-') if k != 'nan'])).to_numpy()

        # Map states to encoded labels
        self.label_to_encoded = mapping_df.set_index('alphabet')['label encoded'].to_dict()
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.info("TCA object initialized successfully")

    def compute_substitution_cost_matrix(self, method='constant', custom_costs=None):
        """
        Compute the substitution cost matrix for the sequences.

        Args:
            method (str): Method to compute the substitution cost matrix ('constant' or 'custom').
            custom_costs (dict): Custom substitution costs (default: None).

        Returns:
            np.ndarray: Substitution cost matrix.
        """
        return compute_substitution_cost_matrix(self.sequences, self.alphabet, method, custom_costs)

    # def optimal_matching(self, seq1, seq2, substitution_cost_matrix, indel_cost=None):
    #    return optimal_matching(seq1, seq2, substitution_cost_matrix, indel_cost, self.alphabet)

    def compute_distance_matrix(self, metric='hamming', substitution_cost_matrix=None):
        """
        Compute the distance matrix for the sequences.

        Args:
            metric (str): Distance metric to use ('hamming', 'euclidean', etc.).
            substitution_cost_matrix (np.ndarray): Substitution cost matrix (default: None).

        Returns:
            np.ndarray: Distance matrix.
        """
        return compute_distance_matrix(self.data, self.sequences, self.label_to_encoded, metric, substitution_cost_matrix, self.alphabet)

    def hierarchical_clustering(self, distance_matrix, method='ward', optimal_ordering=True):
        """
        Perform hierarchical clustering on the distance matrix.

        Args:
            distance_matrix (np.ndarray): Distance matrix.
            method (str): Linkage method for clustering (default: 'ward').
            optimal_ordering (bool): Whether to optimize the leaf order (default: True).

        Returns:
            np.ndarray: Linkage matrix.
        """
        return hierarchical_clustering(self, distance_matrix, method, optimal_ordering)

    def assign_clusters(self, linkage_matrix, num_clusters):
        """
        Assign clusters to the data based on the linkage matrix.

        Args:
            linkage_matrix (np.ndarray): Linkage matrix from hierarchical clustering.
            num_clusters (int): Number of clusters to assign.

        Returns:
            np.ndarray: Cluster assignments.
        """
        return assign_clusters(linkage_matrix, num_clusters)

    def plot_dendrogram(self, linkage_matrix):
        """
        Plot the dendrogram for the hierarchical clustering.

        Args:
            linkage_matrix (np.ndarray): Linkage matrix from hierarchical clustering.
        """
        plot_dendrogram(linkage_matrix)

    def plot_clustermap(self, linkage_matrix):
        """
        Plot a clustermap for the data.

        Args:
            linkage_matrix (np.ndarray): Linkage matrix from hierarchical clustering.
        """
        plot_clustermap(self.data, self.id, self.label_to_encoded, self.colors, self.alphabet, self.states, linkage_matrix)

    def plot_inertia(self, linkage_matrix):
        """
        Plot the inertia diagram to determine the optimal number of clusters.

        Args:
            linkage_matrix (np.ndarray): Linkage matrix from hierarchical clustering.
        """
        plot_inertia(linkage_matrix)

    def plot_cluster_heatmaps(self, clusters, sorted=True):
        return plot_cluster_heatmaps(self.data, self.id, self.label_to_encoded, self.colors, self.alphabet, self.states, clusters, self.leaf_order, sorted)

    def plot_treatment_percentage(self, clusters=None):
        """
        Plot the percentage of patients under each state over time.

        Args:
            clusters (np.ndarray): Cluster assignments for each individual (optional).
        """
        plot_treatment_percentage(self.data, self.id, self.alphabet, self.states, clusters)

    def bar_treatment_percentage(self, clusters=None):
        """
        Plot the percentage of patients under each state over time using bar plots.

        Args:
            clusters (np.ndarray): Cluster assignments for each individual (optional).
        """
        bar_treatment_percentage(self.data, self.id, self.alphabet, self.states, clusters)

####################################### MAIN #######################################
def main():
    """
    Main function to demonstrate the usage of the TCA class.
    """
    df = pd.read_csv('data/dataframe_test.csv')

    # Select relevant columns for analysis
    selected_cols = df[['id', 'month', 'care_status']]

    # Pivot the data to wide format
    pivoted_data = selected_cols.pivot(index='id', columns='month', values='care_status')
    pivoted_data['id'] = pivoted_data.index
    pivoted_data = pivoted_data[['id'] + [col for col in pivoted_data.columns if col != 'id']]

    # Rename columns with a "month_" prefix
    pivoted_data.columns = ['id'] + ['month_' + str(int(col) + 1) for col in pivoted_data.columns[1:]]

    # Select a random sample of 10% of the data
    pivoted_data_random_sample = pivoted_data.sample(frac=0.1, random_state=42).reset_index(drop=True)

    # Filter individuals observed for at least 18 months
    valid_18months_individuals = pivoted_data.dropna(thresh=19).reset_index(drop=True)

    # Select only the first 18 months for analysis
    valid_18months_individuals = valid_18months_individuals[['id'] + [f'month_{i}' for i in range(1, 19)]]

    # Initialize the TCA object
    tca = TCA(data=pivoted_data_random_sample,
              id='id',
              alphabet=['D', 'C', 'T', 'S'],
              states=["diagnostiqué", "en soins", "sous traitement", "inf. contrôlée"])

    # Perform clustering and visualization
    custom_costs = {'D:C': 1, 'D:T': 2, 'D:S': 3, 'C:T': 1, 'C:S': 2, 'T:S': 1}
    substitution_cost_matrix=tca.compute_substitution_cost_matrix(method='custom', custom_costs=custom_costs)
    distance_matrix = tca.compute_distance_matrix(metric='optimal_matching', substitution_cost_matrix=substitution_cost_matrix)
    print("distance matrix :\n", distance_matrix)
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
