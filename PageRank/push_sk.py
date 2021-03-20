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
import time
from sknetwork.data import load_edge_list
from collections import deque

def push(adjacency: Union[sparse.csr_matrix, LinearOperator], seeds: np.ndarray, damping_factor: float,
                 tol: float = 1e-2) -> np.ndarray:

    n = adjacency.shape[0]
    degrees = adjacency.dot(np.ones(n))
    r = np.zeros(n)
    worklist = deque(range(n))

    rev_adjacency = adjacency.transpose().tocsr()

    time_start = time.time()
    for v in range(n):
        # incoming neighbors
        for w in get_neighbors(rev_adjacency, v):
            #print("v:", v, "w:", w)
            r[v] += 1 / degrees[w]
    r *= (1 - damping_factor) * damping_factor
    time_end = time.time()
    print("Prepare residual vector time:", time_end - time_start, "seconds")
    #print(r)

    #seeds = seeds2probs(adjacency.shape[0], seeds)
    #rso = pr.RandomSurferOperator(adjacency, seeds, damping_factor)
    
    scores = np.ones(n) * (1 - damping_factor)

    while len(worklist) > 0:
        #print(len(worklist))
        v = worklist.popleft()
        # scores[v]_new
        scores[v] += r[v]
        for w in get_neighbors(adjacency, v):
            # r_old[w]
            tmp = r[w]
            r[w] += r[v] * (1 - damping_factor) / degrees[v]
            if r[w] >= tol and tmp < tol:
                worklist.append(w)
        r[v] = 0
    norm = np.linalg.norm(scores, 1)  # ||x||_1
    scores /= norm
    return scores


if __name__ == '__main__':
    time_start = time.time()
    graph = load_edge_list('test.txt', directed=True, fast_format=False)
    adjacency = graph.adjacency
    time_end = time.time()
    print("Sknetwork load graph time:", time_end - time_start, "seconds")

    #adjacency = house()

    
    time_start = time.time()
    pagerank = PageRank(damping_factor=0.85)
    scores = pagerank.fit_transform(adjacency)
    time_end = time.time()
    print("Sknetwork power iteration time:", time_end - time_start, "seconds")
    #print(scores)
    

    time_start = time.time()
    scores = push(adjacency, None, 0.85)
    time_end = time.time()
    print("Push Calculation time:", time_end - time_start, "seconds")
    #print(scores)

    