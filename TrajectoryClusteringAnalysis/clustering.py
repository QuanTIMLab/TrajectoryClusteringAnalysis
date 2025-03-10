import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform, pdist
import Levenshtein
from tslearn.metrics import dtw, dtw_path_from_metric, gak
import tqdm
import logging
import timeit
from TrajectoryClusteringAnalysis.optimal_matching import   optimal_matching_fast # Import de la version Cython optimisée


def compute_substitution_cost_matrix(sequences, alphabet, method='constant', custom_costs=None):
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
        vectorized_replace = np.vectorize(label_to_encoded.get)
        return vectorized_replace(sequence)


def compute_distance_matrix(data, sequences, label_to_encoded, metric='hamming', substitution_cost_matrix=None, alphabet=None):
    logging.info(f"Calculating distance matrix using metric: {metric}...")
    start_time = timeit.default_timer()
    n = len(sequences)
    if metric == 'hamming':
        distance_matrix = squareform(np.array(pdist(data.replace(label_to_encoded).drop(columns=['id']), metric=metric)))

    elif metric == 'levenshtein':
        distance_matrix = np.zeros((len(data), len(data)))
        for i in tqdm.tqdm(range(len(sequences))):
            for j in range(i + 1, len(sequences)):
                seq1, seq2 = sequences[i], sequences[j]                  
                distance = Levenshtein.distance(seq1, seq2)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance

    elif metric == 'optimal_matching':
        if substitution_cost_matrix is None:
            logging.error("Substitution cost matrix not found. Please compute the substitution cost matrix first.")
            raise ValueError("Substitution cost matrix not found. Please compute the substitution cost matrix first.")
        distance_matrix = np.zeros((len(data), len(data)))
        print("substitution cost matrix: \n", substitution_cost_matrix)
        print("indel cost: ", np.max(substitution_cost_matrix.values)/2)
        alphabet_dict = {char: i for i, char in enumerate(alphabet)}
        indel_cost = np.max(substitution_cost_matrix.values)/2
        sequences_idx = [np.array([alphabet_dict[s] for s in seq], dtype=np.int32) for seq in sequences]
        for i in tqdm.tqdm(range(len(sequences))):
            for j in range(i + 1, len(sequences)):
                seq1_idx, seq2_idx = sequences_idx[i], sequences_idx[j]
                normalized_dist = optimal_matching_fast(seq1_idx, seq2_idx, substitution_cost_matrix.values, indel_cost=indel_cost)           
                distance_matrix[i, j] = normalized_dist
                distance_matrix[j, i] = normalized_dist

    elif metric == 'dtw':
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
        distance_matrix = np.zeros((len(data), len(data)))
        for i in tqdm.tqdm(range(len(sequences))):
            for j in range(i + 1, len(sequences)):
                seq1, seq2 = replace_labels(sequences[i], label_to_encoded), replace_labels(sequences[j], label_to_encoded)
                distance = dtw_path_from_metric(seq1, seq2, metric=np.abs(seq1 - seq2))
                max_length = max(len(seq1), len(seq2))
                normalized_dist = distance / max_length
                distance_matrix[i, j] = normalized_dist
                distance_matrix[j, i] = normalized_dist

    elif metric == 'gak':
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

def hierarchical_clustering(tca_instance,distance_matrix, method='ward', optimal_ordering=True):
    logging.info(f"Computing the linkage matrix using method: {method}...")
    condensed_distance_matrix = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_distance_matrix, method=method, optimal_ordering=optimal_ordering)
    logging.info("Linkage matrix computed successfully")
    tca_instance.leaf_order = leaves_list(linkage_matrix)
    return linkage_matrix

def assign_clusters(linkage_matrix, num_clusters):
    clusters = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    return clusters