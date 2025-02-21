import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform, pdist
import Levenshtein
from tslearn.metrics import dtw, dtw_path_from_metric, gak
import tqdm
import logging
import timeit


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

def optimal_matching(seq1, seq2, substitution_cost_matrix, indel_cost, alphabet):
    if indel_cost is None:
        indel_cost = max(substitution_cost_matrix.values.flatten()) / 2
    m, n = len(seq1), len(seq2)
    score_matrix = np.zeros((m+1, n+1))
    score_matrix[:, 0] = indel_cost * np.arange(m+1)
    score_matrix[0, :] = indel_cost * np.arange(n+1)

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost_substitute = substitution_cost_matrix.iloc[alphabet.index(seq1[i - 1]), alphabet.index(seq2[j - 1])]
            match = score_matrix[i-1, j-1] + cost_substitute
            delete = score_matrix[i-1, j] + indel_cost
            insert = score_matrix[i, j-1] + indel_cost
            score_matrix[i, j] = min(match, delete, insert)

    optimal_score = score_matrix[m, n]
    return optimal_score

def replace_labels(sequence, label_to_encoded):
        vectorized_replace = np.vectorize(label_to_encoded.get)
        return vectorized_replace(sequence)


def compute_distance_matrix(data, sequences, label_to_encoded, metric='hamming', substitution_cost_matrix=None, alphabet=None):
    logging.info(f"Calculating distance matrix using metric: {metric}...")
    start_time = timeit.default_timer()

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
        print("indel cost: ", max(substitution_cost_matrix.values.flatten()) / 2)
        for i in tqdm.tqdm(range(len(sequences))):
            for j in range(i + 1, len(sequences)):
                seq1, seq2 = sequences[i], sequences[j]
                distance = optimal_matching(seq1, seq2, substitution_cost_matrix, indel_cost=None, alphabet=alphabet)
                max_length = max(len(seq1), len(seq2))
                normalized_dist = distance / max_length
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