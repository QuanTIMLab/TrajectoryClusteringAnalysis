"""
Module pour les algorithmes de clustering de trajectoires.

Ce module contient des fonctions pour calculer les matrices de substitution,
les matrices de distances, et effectuer un clustering hiérarchique.
"""


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings from pandas
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform, pdist, cityblock
import Levenshtein
from tslearn.metrics import dtw, dtw_path_from_metric, gak
import tqdm
import logging
import timeit
from trajectoryclusteringanalysis.optimal_matching import optimal_matching_fast # Import de la version Cython optimisée
import kmedoids


def compute_substitution_cost_matrix(sequences, alphabet, method='constant', custom_costs=None):
    """
    Calcule une matrice de coûts de substitution pour les séquences.

    Args:
        sequences (list): Liste des séquences à analyser.
        alphabet (list): Liste des états possibles.
        method (str): Méthode pour calculer les coûts ('constant', 'custom', 'frequency').
        custom_costs (dict): Coûts personnalisés pour les substitutions (optionnel).

    Returns:
        pd.DataFrame: Matrice de coûts de substitution.
    """
    num_states = len(alphabet)
    substitution_matrix = np.zeros((num_states, num_states))

    if method == 'constant':
        for i in range(num_states):
            for j in range(num_states):
                if i != j:
                    substitution_matrix[i, j] = 2

    elif method == 'custom':
        for i in range(num_states):
            for j in range(num_states):
                if i != j:
                    state_i = alphabet[i]
                    state_j = alphabet[j]
                    try:
                        key = state_i + ':' + state_j
                        cost = custom_costs[key]
                    except:
                        key = state_j + ':' + state_i  
                        cost = custom_costs[key]
                    substitution_matrix[i, j] = cost

    elif method == 'frequency':
        substitution_frequencies = np.zeros((num_states, num_states))

        for sequence in sequences:
            sequence = [char if char != 'nan' else '-' for char in sequence.split('-')]
            for i in range(len(sequence) - 1):
                state_i = alphabet.index(sequence[i])
                state_j = alphabet.index(sequence[i + 1])
                substitution_frequencies[state_i, state_j] += 1

        substitution_probabilities = substitution_frequencies / substitution_frequencies.sum(axis=1, keepdims=True)

        for i in range(num_states):
            for j in range(num_states):
                if i != j:
                    substitution_matrix[i, j] = 2 - substitution_probabilities[i, j] - substitution_probabilities[j, i]

    substitution_cost_matrix = pd.DataFrame(substitution_matrix, index=alphabet, columns=alphabet)
    return substitution_cost_matrix

def replace_labels(sequence, label_to_encoded):
    """
    Replaces sequence labels with their encoded values.

    Parameters:
    - sequence: Sequence to be encoded.
    - label_to_encoded: Dictionary mapping labels to encoded values.

    Returns:
    - Encoded sequence.
    """
    try:
        vectorized_replace = np.vectorize(label_to_encoded.__getitem__)
        return vectorized_replace(sequence)
    except KeyError as e:
        raise ValueError(f"Label {e} not found in label_to_encoded mapping.")
    #vectorized_replace = np.vectorize(label_to_encoded.get)
    #return vectorized_replace(sequence)

def compute_distance_matrix(data, sequences, label_to_encoded, metric='hamming', substitution_cost_matrix=None, alphabet=None, indel_cost=None, id=None):
    """
    Calcule une matrice de distances entre les séquences.

    Args:
        data (pd.DataFrame): Données d'entrée.
        sequences (list): Liste des séquences.
        label_to_encoded (dict): Mapping des labels vers des valeurs encodées.
        metric (str): Métrique de distance ('hamming', 'levenshtein', etc.).
        substitution_cost_matrix (pd.DataFrame): Matrice de coûts de substitution (optionnel).
        alphabet (list): Liste des états possibles (optionnel).

    Returns:
        np.ndarray: Matrice de distances.
    """
    logging.info(f"Calculating distance matrix using metric: {metric}...")
    start_time = timeit.default_timer()
    if sequences is None:
        pass
    else: 
        n = len(sequences)
    
    if metric == 'euclidean':   
        # Compute Euclidean distance
        if id in data.columns:
            data = data.drop(columns=[id])  # Drop the 'id' column if it exists
        distance_matrix = pdist(data, metric=metric)

    elif metric == 'hamming':
        # Compute Hamming distance
        distance_matrix = squareform(np.array(pdist(data.replace(label_to_encoded).drop(columns=['id']), metric=metric)))

    elif metric == 'levenshtein':
        # Compute Levenshtein distance
        distance_matrix = np.zeros((len(data), len(data)))
        for i in tqdm.tqdm(range(len(sequences))):
            for j in range(i + 1, len(sequences)):
                seq1, seq2 = sequences[i], sequences[j]                  
                distance = Levenshtein.distance(seq1, seq2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    elif metric == 'optimal_matching':
        # Compute Optimal Matching distance
        if substitution_cost_matrix is None:
            logging.error("Substitution cost matrix not found. Please compute the substitution cost matrix first.")
            raise ValueError("Substitution cost matrix not found. Please compute the substitution cost matrix first.")
        distance_matrix = np.zeros((len(data), len(data)))
         # Ensure substitution_cost_matrix is a NumPy array for .values access
        sub_matrix_values = substitution_cost_matrix.values if isinstance(substitution_cost_matrix, pd.DataFrame) else substitution_cost_matrix
        
        current_indel_cost = indel_cost
        if current_indel_cost is None:
            # Heuristic: indel cost is half the max substitution cost
            current_indel_cost = np.max(sub_matrix_values) / 2
            print(f"Calculated indel cost: {current_indel_cost} (from max sub_cost: {np.max(sub_matrix_values)})")
        else:
            print(f"User-provided indel cost: {current_indel_cost}")

        print("substitution cost matrix: \n", substitution_cost_matrix)
        alphabet_dict = {char: i for i, char in enumerate(alphabet)}
        #indel_cost = np.max(substitution_cost_matrix.values)/2
        sequences_idx = [np.array([alphabet_dict[s] for s in seq], dtype=np.int32) for seq in sequences]
        for i in tqdm.tqdm(range(len(sequences))):
            for j in range(i + 1, len(sequences)):
                seq1_idx, seq2_idx = sequences_idx[i], sequences_idx[j]
                normalized_dist = optimal_matching_fast(seq1_idx, seq2_idx, sub_matrix_values, indel_cost=current_indel_cost)           
                distance_matrix[i, j] = normalized_dist
                distance_matrix[j, i] = normalized_dist

    elif metric == 'dtw':
        # Compute Dynamic Time Warping (DTW) distance
        distance_matrix = np.zeros((len(data), len(data)))
        for i in tqdm.tqdm(range(len(sequences))):
            for j in range(i + 1, len(sequences)):
                seq1, seq2 = replace_labels(sequences[i], label_to_encoded), replace_labels(sequences[j], label_to_encoded)
                distance = dtw(seq1, seq2)
                max_length = max(len(seq1), len(seq2))
                normalized_dist = distance / max_length
                distance_matrix[i, j] = normalized_dist
                distance_matrix[j, i] = normalized_dist

    elif metric == 'dtw_path_from_metric':
        # Compute DTW distance using a custom metric
        distance_matrix = np.zeros((len(data), len(data)))
        for i in tqdm.tqdm(range(len(sequences))):
            for j in range(i + 1, len(sequences)):
                seq1, seq2 = replace_labels(sequences[i], label_to_encoded), replace_labels(sequences[j], label_to_encoded)
                #distance = dtw_path_from_metric(seq1, seq2, metric=np.abs(seq1 - seq2))
                distance, _ = dtw_path_from_metric(seq1, seq2, metric=cityblock)
                max_length = max(len(seq1), len(seq2))
                normalized_dist = distance / max_length
                distance_matrix[i, j] = normalized_dist
                distance_matrix[j, i] = normalized_dist

    elif metric == 'gak':
        # Compute Global Alignment Kernel (GAK) distance
        distance_matrix = np.zeros((len(data), len(data)))
        for i in tqdm.tqdm(range(len(sequences))):
            for j in range(i + 1, len(sequences)):
                seq1, seq2 = replace_labels(sequences[i], label_to_encoded), replace_labels(sequences[j], label_to_encoded)
                distance = gak(seq1, seq2)
                max_length = max(len(seq1), len(seq2))
                normalized_dist = distance / max_length
                distance_matrix[i, j] = normalized_dist
                distance_matrix[j, i] = normalized_dist

    c_time = timeit.default_timer() - start_time
    logging.info(f"Time taken for computation: {c_time:.2f} seconds")
    assert(np.allclose(distance_matrix, distance_matrix.T)), "Distance matrix is not symmetric"
    return distance_matrix

def hierarchical_clustering(tca_instance, distance_matrix, method='ward', optimal_ordering=True):
    """
    Effectue un clustering hiérarchique sur une matrice de distances.

    Args:
        tca_instance (TCA): Instance de la classe TCA.
        distance_matrix (np.ndarray): Matrice de distances.
        method (str): Méthode de linkage ('ward', 'single', etc.).
        optimal_ordering (bool): Optimiser l'ordre des feuilles (par défaut: True).

    Returns:
        np.ndarray: Matrice de linkage.
    """
    logging.info(f"Computing the linkage matrix using method: {method}...")
    condensed_distance_matrix = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distance_matrix, method=method, optimal_ordering=optimal_ordering)
    logging.info("Linkage matrix computed successfully")
    tca_instance.leaf_order = leaves_list(linkage_matrix)
    return linkage_matrix

def assign_clusters(linkage_matrix, num_clusters):
    """
    Assigne des étiquettes de clusters aux données.

    Args:
        linkage_matrix (np.ndarray): Matrice de linkage.
        num_clusters (int): Nombre de clusters à assigner.

    Returns:
        np.ndarray: Étiquettes des clusters.
    """
    clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    return clusters

def k_medoids_clustering_faster(distance_matrix, num_clusters, method='fasterpam', init='random', max_iter=300, random_state=None, **kwargs):
    '''
    Performs K-Medoids clustering on a precomputed distance matrix.

    Args:
        distance_matrix (np.ndarray): Square distance matrix.
        num_clusters (int): The desired number of clusters.
        method (str, optional): The method to use for KMedoids.
            "fasterpam" (default), "fastpam1", "pam", "alternate", "fastermsc", "fastmsc", "pamsil" or "pammedsil".
        init (string, "random" (default), "first" or "build") – initialization method.
        max_iter (int, optional): Maximum number of iterations. Defaults to 300.
        random_state (int, RandomState instance or None, optional):
            Determines random number generation for centroid initialization.
            Use an int to make the randomness deterministic. Defaults to None.
        **kwargs: Additional keyword arguments to pass to KMedoids.

    Returns:
        kmedoids.KMedoids: the results containing:
            - cluster_centers : None for 'precomputed'
            - medoid_indices : The indices of the medoid rows in X.
            - labels : Labels of each point.
            - inertia : Sum of distances of samples to their closest cluster center.
    '''
    if not isinstance(distance_matrix, np.ndarray):
        raise TypeError("distance_matrix must be a NumPy array.")
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        distance_matrix = squareform(distance_matrix)  # Ensure it's a square matrix
    if not isinstance(num_clusters, int) or num_clusters <= 0: # Ensure num_clusters is positive int
        raise ValueError("num_clusters must be a positive integer.")
    if num_clusters > distance_matrix.shape[0]:
        raise ValueError("num_clusters cannot be greater than the number of samples.")

    logging.info(f"Performing K-Medoids clustering with {num_clusters} clusters...")
    kmedoids_model = kmedoids.KMedoids(
        n_clusters=num_clusters,
        metric='precomputed',  # Crucial for using a distance matrix
        method=method,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
        **kwargs
    ).fit(distance_matrix)  # Fit the model to the distance matrix
    # KMedoids expects the distance matrix itself as input to fit/fit_predict
    labels = kmedoids_model.labels_ + 1  # Adjust labels to start from 1
    medoid_indices = kmedoids_model.medoid_indices_
    inertia = kmedoids_model.inertia_
    logging.info("K-Medoids clustering completed.")
    return labels, medoid_indices, inertia