from typing import Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs, LinearOperator, bicgstab
import sknetwork.linalg.ppr_solver as pr
from sknetwork.data import house
from sknetwork.ranking import PageRank
from sknetwork.utils.check import check_format, check_square, check_damping_factor
from sknetwork.utils.seeds import seeds2probs, stack_seeds
from sknetwork.utils import get_neighbors

def push(adjacency: Union[sparse.csr_matrix, LinearOperator], seeds: np.ndarray, damping_factor: float,
                 tol: float = 1e-6) -> np.ndarray:

    n = adjacency.shape[0]
    degrees = adjacency.dot(np.ones(n))
    r = np.zeros(n)
    worklist = set(range(n))

    for v in range(n):
        for w in get_neighbors(adjacency, v):
            r[v] += 1 / degrees[w]
        r[v] *= (1 - damping_factor) * damping_factor

    #seeds = seeds2probs(adjacency.shape[0], seeds)
    #rso = pr.RandomSurferOperator(adjacency, seeds, damping_factor)
    
    scores = np.ones(n) * (1 - damping_factor)

    while len(worklist) > 0:
        v = worklist.pop()
        # scores[v]_new
        scores[v] += r[v]
        for w in get_neighbors(adjacency, v):
            # r_old[w]
            tmp = r[w]
            r[w] += r[v] * damping_factor / degrees[v]
            if r[w] >= tol and tmp < tol:
                worklist.add(w)
        r[v] = 0
    norm = np.linalg.norm(scores, 1)  # ||x||_1
    scores /= norm
    return scores


if __name__ == '__main__':
    adjacency = house()
    scores = push(adjacency, None, 0.85)
    print("Push:")
    print(scores)
    pagerank = PageRank(damping_factor=0.85)
    scores = pagerank.fit_transform(adjacency)
    print("Sknetwork power iteration:")
    print(scores)