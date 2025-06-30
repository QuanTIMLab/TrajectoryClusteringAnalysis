import numpy as np
import timeit
import tracemalloc

from TrajectoryClusteringAnalysis.optimal_matching import optimal_matching_fast as cython_fast

from numba import njit

@njit
def optimal_matching_numba(seq1, seq2, substitution_cost_matrix, indel_cost):
    m, n = len(seq1), len(seq2)
    score_matrix = np.zeros((m+1, n+1))
    score_matrix[:, 0] = indel_cost * np.arange(m+1)
    score_matrix[0, :] = indel_cost * np.arange(n+1)

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost_substitute = substitution_cost_matrix[seq1[i-1], seq2[j-1]]
            match = score_matrix[i-1, j-1] + cost_substitute
            delete = score_matrix[i-1, j] + indel_cost
            insert = score_matrix[i, j-1] + indel_cost
            score_matrix[i, j] = min(match, delete, insert)

    return score_matrix[m, n]/max(m, n)

# Générer des séquences aléatoires
seq1 = np.random.randint(0, 5, 1000)
seq2 = np.random.randint(0, 5, 1000)
substitution_cost_matrix = np.random.rand(5, 5)
indel_cost = 1.0

# Fonction pour mesurer le temps d'exécution
def measure_time(func, *args):
    tracemalloc.start()
    start_mem = tracemalloc.get_traced_memory()[1]
    start = timeit.default_timer()
    result = func(*args)
    end = timeit.default_timer()
    end_mem = tracemalloc.get_traced_memory()[1]  # Mémoire finale
    tracemalloc.stop()  # Arrêt du tracking
    return result, end - start,end_mem - start_mem 
# Benchmark de la version Cython avec allocation manuelle
distance_cython_fast, time_cython_fast,mem_used  = measure_time(cython_fast, seq1, seq2, substitution_cost_matrix, indel_cost)
# Benchmark de la version Numba

distance_numba, time_numba,mem_used  = measure_time(optimal_matching_numba, seq1, seq2, substitution_cost_matrix, indel_cost)
# Affichage des résultats
print("Résultats du benchmarking : deux séquences de taille 100")
print(f"Version Numba : Distance = {distance_numba}, Temps = {time_numba:.6f} s, Mémoire: {mem_used} bytes")
print(f"Version Cython Fast (Mémoire Manuelle) : Distance = {distance_cython_fast}, Temps = {time_cython_fast:.6f} s, Mémoire: {mem_used} bytes")
