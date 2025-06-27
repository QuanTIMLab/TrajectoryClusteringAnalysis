from libc.stdlib cimport malloc, free
from libc.math cimport INFINITY
cpdef double optimal_matching_fast(const int[:] seq1, 
                                  const int[:] seq2, 
                                  const double[:, :] substitution_cost_matrix, 
                                  double indel_cost) nogil:
    """ Version optimisée d'Optimal Matching en Cython """
    cdef int m = seq1.shape[0]
    cdef int n = seq2.shape[0]
    cdef int i, j
    cdef double cost_substitute, match, delete, insert
    cdef int idx_i_j, idx_im1_j, idx_i_jm1, idx_im1_jm1
    # Allocation manuelle de la matrice de coût
    cdef double* score_matrix = <double*> malloc((m + 1) * (n + 1) * sizeof(double))
    if not score_matrix:
        with gil:
            raise MemoryError("Allocation de la mémoire échouée !")

    for i in range(m + 1):
        score_matrix[i * (n + 1)] = i * indel_cost
    for j in range(n + 1):
        score_matrix[j] = j * indel_cost

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost_substitute = substitution_cost_matrix[seq1[i - 1], seq2[j - 1]]
            idx_i_j = i * (n + 1) + j
            idx_im1_j = (i - 1) * (n + 1) + j
            idx_i_jm1 = i * (n + 1) + (j - 1)
            idx_im1_jm1 = (i - 1) * (n + 1) + (j - 1)
            match = score_matrix[idx_im1_jm1] + cost_substitute
            delete = score_matrix[idx_im1_j] + indel_cost
            insert = score_matrix[idx_i_jm1] + indel_cost
            score_matrix[idx_i_j] = min(match, delete, insert)
    cdef double result = score_matrix[m * (n + 1) + n]
    free(score_matrix)
    if m == 0 and n == 0:
        return 0.0
    #return result
    return result / max(m, n)  # Normalisation


