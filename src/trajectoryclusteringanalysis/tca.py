import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings from pandas
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder
from src.trajectoryclusteringanalysis.unidimensional.clustering import *
from src.trajectoryclusteringanalysis.multidimensional.analysis import *
from src.trajectoryclusteringanalysis.plotting import *

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
    def __init__(self, data, index_col, time_col=None, event_col=None, alphabet=None, states=None, mode=None, colors='viridis'):
        """
        Initialize the TCA object with input data and parameters.

        Args:
            data (pd.DataFrame): Either a DataFrame in long format with columns for id, time, and event, or a DataFrame in wide format with individuals as rows and time points as columns.
            id (str): Column name representing unique identifiers for individuals.
            alphabet (list): List of possible states in the trajectories.
            states (list): Descriptive labels for the states.
            mode (str): Mode of analysis ('unidimensional' vs 'multidimensional').
            colors (str): Colormap for visualizations (default: 'viridis').
        """
        self.data = data
        self.index_col = index_col
        self.time_col = time_col
        self.event_col = event_col
        self.alphabet = alphabet
        self.states = states
        self.colors = colors
        self.mode = mode
        self.leaf_order = None
        self.substitution_cost_matrix = None
        self.sequences = None
        logging.basicConfig(level=logging.INFO)

        # Validate input data regarding the mode
        if self.mode == 'unidimensional':
            assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
            assert self.data.shape[1] > 1, "data must have more than one column, in a wide format"
            assert self.index_col in self.data.columns, f"{self.index_col} must be a column in the data"
            assert len(self.data[self.index_col].unique()) > 0, "data must contain at least one unique individual"
            assert self.data[self.index_col].duplicated().sum() == 0, "There are duplicates in the data. Your dataset must be in wide format"

            # Prepare unidimensional data for TCA
            data_ready_for_TCA = self.data.copy()
            data_ready_for_TCA['Sequence'] = data_ready_for_TCA.drop(self.index_col, axis=1).apply(lambda x: '-'.join(x.astype(str)), axis=1)
            data_ready_for_TCA = data_ready_for_TCA[['id', 'Sequence']]
            self.sequences = data_ready_for_TCA['Sequence'].apply(lambda x: np.array([k for k in x.split('-') if k != 'nan'])).to_numpy()
            
        elif self.mode == 'multidimensional':
            assert isinstance(data, pd.DataFrame), "data must be a pandas DataFrame"
            assert MultidimensionalAnalyzer(self.data, self.index_col, self.time_col, self.event_col).has_time_event_structure(), "data must have a time-event structure with columns for id, time, and event"
            assert self.index_col in self.data.columns, f"{self.index_col} must be a column in the data"
            assert self.time_col in self.data.columns, f"{self.time_col} must be a column in the data"
            assert self.event_col in self.data.columns, f"{self.event_col} must be a column in the data"
            assert len(data[self.index_col].unique()) > 0, "data must contain at least one unique individual"

            self.colors = 'Spectral_r'

        else:
            raise ValueError("mode must be either 'unidimensional' or 'multidimensional'")

        # Print dataset information
        print("Dataset :")
        print("data shape: ", self.data.shape)
        if self.alphabet is not None and self.states is not None:
            mapping_df = pd.DataFrame({'alphabet': self.alphabet, 'label': self.states, 'label encoded': range(1, 1 + len(self.alphabet))})
            print("state coding:\n", mapping_df)
            # Map states to encoded labels
            self.label_to_encoded = mapping_df.set_index('alphabet')['label encoded'].to_dict()

        else:
            if self.event_col is not None:
                self.label_encoder = LabelEncoder()
                self.data[self.event_col] = self.label_encoder.fit_transform(self.data[self.event_col])
                self.label_to_encoded = dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))
                mapping_df = pd.DataFrame({'label': list(self.label_to_encoded.keys()), 'label encoded': list(self.label_to_encoded.values())})
                self.alphabet = list(self.label_to_encoded.keys())
                print("state coding:\n", mapping_df)
            else:
                logging.warning("Alphabet and states are not provided. The TCA object will not be able to compute substitution costs or distance matrices.")    

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.info("TCA object initialized successfully")
        logging.info(f"The {self.mode} analysis mode is set. TCA object will analyze the data accordingly.")

    def get_label(self, label_encoded):
        """
        Get the label corresponding to the encoded label.

        Args:
            label_encoded (int): Encoded label.

        Returns:
            str: Corresponding label.
        """
        if hasattr(self, 'label_to_encoded'):
            for key, value in self.label_to_encoded.items():
                if value == label_encoded:
                    return key
        else:
            raise ValueError("Label encoding is not available. Ensure that alphabet and states are provided.")

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

    def compute_distance_matrix(self, data, metric='hamming', substitution_cost_matrix=None, indel_cost=None):
        """
        Compute the distance matrix for the sequences.

        Args:
            metric (str): Distance metric to use ('hamming', 'euclidean', etc.).
            substitution_cost_matrix (np.ndarray): Substitution cost matrix (default: None).

        Returns:
            np.ndarray: Distance matrix.
        """
        return compute_distance_matrix(data, self.sequences, self.label_to_encoded, metric, substitution_cost_matrix, self.alphabet, indel_cost, id=self.index_col)

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
    
    def kmedoids_clustering(self, distance_matrix, num_clusters=4, method='fasterpam', init='random', max_iter=300, random_state=None, **kwargs):
        '''
        Performs K-Medoids clustering on a precomputed distance matrix.

        This method wraps the k_medoids_clustering function from clustering.py.

        Args:
            num_clusters (int): The desired number of clusters.
            distance_matrix (np.ndarray): A precomputed square distance matrix.
                                          This matrix must be computed beforehand, e.g., using
                                          TCA.compute_distance_matrix().
            method (str, optional): The KMedoids method ( "fasterpam" (default), "fastpam1", "pam", "alternate", "fastermsc", "fastmsc", "pamsil" or "pammedsil")
            init (string, "random" (default), "first" or "build") – initialization method.
            max_iter (int, optional): Maximum number of iterations for KMedoids. Defaults to 300.
            random_state (int, RandomState instance or None, optional):
                Determines random number generation for KMedoids. Defaults to None.
            **kwargs: Additional keyword arguments passed to KMedoids.

        Returns:
             kmedoids.KMedoids: the results containing:
                - cluster_centers : None for 'precomputed'
                - medoid_indices : The indices of the medoid rows in X.
                - labels : Labels of each point.
                - inertia : Sum of distances of samples to their closest cluster center.
        '''
        if distance_matrix is None:
            raise ValueError("A precomputed distance_matrix must be provided.")
        

        return k_medoids_clustering_faster(
            distance_matrix,
            num_clusters,
            method=method,
            init=init,
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )
        
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
        plot_clustermap(self.data, self.index_col, self.label_to_encoded, self.colors, self.alphabet, self.states, linkage_matrix)

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
    
    def plot_filtered_heatmap(self,labels=None, linkage_matrix=None, kernel_size=(10, 7)):
        """
        Plot a heatmap of patient treatment sequences, optionally filtered using a modal filter.

        Reordering of the sequences is based on the clustering method used:
        - If K-Medoids was used, provide the cluster labels via `labels`.
        - If Hierarchical Clustering (CAH) was used, provide the linkage matrix via `linkage_matrix`.

        Parameters:
        - labels (np.ndarray, optional): Cluster labels from K-Medoids clustering.
        - linkage_matrix (np.ndarray, optional): Linkage matrix from hierarchical clustering.
        - kernel_size (tuple of int, optional): Size of the modal filter kernel (rows, cols).
                                                Use (0, 0) to disable filtering. Default is (10, 7).

        Returns:
        - None. Displays a heatmap using matplotlib
        """
        if (labels is None and linkage_matrix is None) or (labels is not None and linkage_matrix is not None):
            raise ValueError("You must provide exactly one of 'labels' (K-Medoids) or 'linkage_matrix' (CAH).")
        plot_filtered_heatmap(self.data, self.id, self.label_to_encoded, self.colors, self.alphabet, self.states,labels=labels, linkage_matrix=linkage_matrix,kernel_size=kernel_size)

####################################### MAIN #######################################
def main():
    """
    Main function to demonstrate the usage of the TCA class.
    """

    ###### UNIDIMENSIONAL ANALYSIS ######
    # df = pd.read_csv('data/dataframe_test.csv')

    # selected_cols = df[['id', 'month', 'care_status']]

    # # Pivot the data to wide format
    # pivoted_data = selected_cols.pivot(index='id', columns='month', values='care_status')
    # pivoted_data['id'] = pivoted_data.index
    # pivoted_data = pivoted_data[['id'] + [col for col in pivoted_data.columns if col != 'id']]
    # pivoted_data.columns = ['id'] + ['month_' + str(int(col) + 1) for col in pivoted_data.columns[1:]]

    # pivoted_data_random_sample = pivoted_data.sample(frac=0.1, random_state=42).reset_index(drop=True)

    # valid_18months_individuals = pivoted_data.dropna(thresh=19).reset_index(drop=True)
    # valid_18months_individuals = valid_18months_individuals[['id'] + [f'month_{i}' for i in range(1, 19)]]

    # # Initialize the TCA object
    # tca = TCA(data=pivoted_data_random_sample,
    #           id='id',
    #           alphabet=['D', 'C', 'T', 'S'],
    #           states=["diagnostiqué", "en soins", "sous traitement", "inf. contrôlée"])

    # # Perform clustering and visualization
    # custom_costs = {'D:C': 1, 'D:T': 2, 'D:S': 3, 'C:T': 1, 'C:S': 2, 'T:S': 1}
    # substitution_cost_matrix=tca.compute_substitution_cost_matrix(method='custom', custom_costs=custom_costs)
    # distance_matrix = tca.compute_distance_matrix(metric='optimal_matching', substitution_cost_matrix=substitution_cost_matrix)
    # print("distance matrix :\n", distance_matrix)
    # linkage_matrix = tca.hierarchical_clustering(distance_matrix)
    # tca.plot_dendrogram(linkage_matrix)
    # tca.plot_clustermap(linkage_matrix)
    # tca.plot_inertia(linkage_matrix)
    # clusters = tca.assign_clusters(linkage_matrix, num_clusters=4)
    # tca.plot_cluster_heatmaps(clusters, sorted=True)
    # tca.plot_treatment_percentage()
    # tca.plot_treatment_percentage(clusters)
    # tca.bar_treatment_percentage()
    # tca.bar_treatment_percentage(clusters)
    # tca.plot_filtered_heatmap(linkage_matrix=linkage_matrix, kernel_size=(0, 0))  # Pas de filtre modal
    # tca.plot_filtered_heatmap(linkage_matrix=linkage_matrix, kernel_size=(10, 7)) 



    ###### MULTIDIMENSIONAL ANALYSIS ######
    df = pd.read_excel('data/multidimensional_data.xlsx')
    print(df.head())

    tca = TCA(data=df,
              index_col='ID_PATIENT',
              time_col='Months_Since_First_Events',
              event_col='Lib_traitement',
              mode='multidimensional')
    
    analyzer = MultidimensionalAnalyzer(tca.data, tca.index_col, tca.time_col, tca.event_col)
    analyzer.transform_time_event_structure_to_tensor()
    print("Tensor shape:", analyzer.get_tensor_shape())
    tensor = analyzer.get_tensor()
    
    # Example decomposition
    rank = 5
    time_window_length = 3
    reg_term_ns = 0.5
    reg_term_s = 0.5
    metric = 'Bernoulli'
    learning_rate = 1e-2
    n_epochs = 10

    unique_patients = df['ID_PATIENT'].unique()
    patient_index = np.where(unique_patients == 1102101064)[0][0]  

    plot_input_matrix(tensor, id=patient_index, labels=tca.label_to_encoded)

    analyzer.fit_swotted_decomposition(tensor, rank, time_window_length, reg_term_ns, reg_term_s, metric, learning_rate, n_epochs)
    analyzer.get_decomposition_results(labels=tca.label_to_encoded, id=patient_index)

    ph_intensity = analyzer.to_phenotype_intensity()
    distance_matrix = tca.compute_distance_matrix(data=ph_intensity, metric='euclidean')
    print("Distance matrix:\n", distance_matrix)
    linkage_matrix = tca.hierarchical_clustering(distance_matrix)
    tca.plot_dendrogram(linkage_matrix)
    tca.plot_clustermap(linkage_matrix)
    


if __name__ == "__main__":
    main()
