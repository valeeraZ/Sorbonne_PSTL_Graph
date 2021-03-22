# distutils: language = c++
# cython: language_level=3
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import numpy as np
cimport numpy as np
cimport cython
from libcpp.queue cimport queue
from cython.parallel cimport prange

@cython.boundscheck(False)
@cython.wraparound(False)
#@cython.cdivision(True)
def push_pagerank(int n, np.ndarray[np.int32_t, ndim=1] degrees, int[:] indptr, int[:] indices, int[:] rev_indptr, int[:] rev_indices, float damping_factor, float tol):
    cdef np.ndarray[np.float32_t, ndim=1] r
    cdef int v
    cdef int w
    cdef int j1
    cdef int j2
    cdef int jj
    cdef int[:] indexes
    cdef int index
    cdef queue[int] worklist
    cdef np.ndarray[np.float32_t, ndim=1] scores
    cdef float tmp
    cdef float norm

    r = np.zeros(n, dtype=np.float32)
    for v in prange(n, nogil=True):
        j1 = rev_indptr[v]
        j2 = rev_indptr[v+1]
        for jj in range(j1, j2):
            w = rev_indices[jj]
            r[v] += 1 / degrees[w]
        """
        for w in rev_indices[rev_indptr[v]:rev_indptr[v+1]]:
            r[v] += 1 / degrees[w]
        """
        r[v] *= (1 - damping_factor) * damping_factor

    # node with high residual value will be processed first
    indexes = np.argsort(-r).astype(np.int32)
    for index in indexes:
        worklist.push(index)

    scores = np.full(n, (1-damping_factor), dtype=np.float32)

    while not worklist.empty():
        v = worklist.front()
        worklist.pop()
        # scores[v]_new
        scores[v] += r[v]
        # iterate node v's out-coming neighbors w
        j1 = indptr[v]
        j2 = indptr[v + 1]
        for jj in prange(j1, j2, nogil=True):
            w = indices[jj]
            # r_old[w]
            tmp = r[w]
            r[w] += r[v] * (1 - damping_factor) / degrees[v]
            if r[w] >= tol > tmp:
                worklist.push(w)
        """
        for w in indices[j1:j2]:
            tmp = r[w]
            r[w] += r[v] * (1 - damping_factor) / degrees[v]
            if r[w] >= tol > tmp:
                worklist.push(w)
        r[v] = 0
        """
    norm = np.linalg.norm(scores, 1)  # ||x||_1
    scores /= norm
    return scores

