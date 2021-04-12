from push import push_pagerank
import pandas as pd
import time
import numpy as np
from scipy import sparse
from sknetwork.data import load_edge_list, house, miserables, karate_club
from sknetwork.ranking import PageRank
from read_graph import print_highest_lowest_values
from sknetwork.utils.seeds import seeds2probs

# load graph

adjacency = house()

# read names of pages

# pages = pd.read_table('data/pageNum.txt', encoding='utf-8', header=None).values.tolist()


# parameters

seeds = None

seeds = seeds2probs(adjacency.shape[0], seeds)
tol = 10
damping_factor = 0.85

# beginning push algorithm
time_start = time.time()
n = adjacency.shape[0]
degrees = adjacency.dot(np.ones(n)).astype(np.int32)
rev_adjacency = adjacency.transpose().tocsr()

indptr = adjacency.indptr.astype(np.int32)
indices = adjacency.indices.astype(np.int32)
rev_indptr = rev_adjacency.indptr.astype(np.int32)
rev_indices = rev_adjacency.indices.astype(np.int32)

scores = push_pagerank(n, degrees, indptr, indices, rev_indptr, rev_indices, seeds.astype(np.float32), damping_factor, tol)
time_end = time.time()
print("Push Calculation time:", time_end - time_start, "seconds")
