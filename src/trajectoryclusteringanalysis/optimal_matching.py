"""Pure Python fallback for Optimal Matching.

This module mirrors the Cython API so the package works even when the
compiled extension is unavailable.
"""

from __future__ import annotations

import numpy as np


def optimal_matching_fast(seq1, seq2, substitution_cost_matrix, indel_cost):
    """Compute normalized Optimal Matching distance between two encoded sequences.

    Parameters
    ----------
    seq1, seq2 : array-like of int
        Encoded sequences.
    substitution_cost_matrix : array-like of shape (n_states, n_states)
        Substitution costs between states.
    indel_cost : float
        Insertion/deletion cost.

    Returns
    -------
    float
        Optimal Matching distance normalized by max(len(seq1), len(seq2)).
    """
    s1 = np.asarray(seq1, dtype=np.int32)
    s2 = np.asarray(seq2, dtype=np.int32)
    sub = np.asarray(substitution_cost_matrix, dtype=np.float64)

    m = int(s1.shape[0])
    n = int(s2.shape[0])

    if m == 0 and n == 0:
        return 0.0

    score = np.zeros((m + 1, n + 1), dtype=np.float64)
    score[:, 0] = np.arange(m + 1, dtype=np.float64) * float(indel_cost)
    score[0, :] = np.arange(n + 1, dtype=np.float64) * float(indel_cost)

    for i in range(1, m + 1):
        si = s1[i - 1]
        for j in range(1, n + 1):
            sj = s2[j - 1]
            match = score[i - 1, j - 1] + sub[si, sj]
            delete = score[i - 1, j] + indel_cost
            insert = score[i, j - 1] + indel_cost
            score[i, j] = min(match, delete, insert)

    return float(score[m, n] / max(m, n))
