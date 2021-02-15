from read_graph import readGraph, makeAdjArray, print_highest_lowest_values
import sys
import numpy as np
import time


def topologyPageRank(edges, pages, alpha, adj_array, node_to_index):
    time_start = time.time()
    # Number of nodes
    n = len(pages)
    # Number of edges
    e = len(edges)
    # Page Rank vector
    x = np.ones(n) * (1 - alpha)
    # precision, variant of difference inaccuracy between two iterations
    eps = 0.000001
    # nodes, in fact the indexes in an array, not ID in file
    nodes = node_to_index.values()
    print("Calculating Page Rank value with alpha =", alpha)

    degree_out = np.zeros(n)
    for node in adj_array:
        degree_out[node] = len(adj_array.get(node))

    # S, T of each node
    t = adj_array
    s = {}
    for node in adj_array:
        out_neighbours = adj_array.get(node)
        for out_node in out_neighbours:
            if out_node not in s.keys():
                s[out_node] = [node]
            else:
                s[out_node].append(node)
    # save the difference of 2 iterations
    delta = np.zeros(n)
    while True:
        for v in nodes:
            tmp = x
            sum_value = 0
            if v in s.keys():
                for w in s[v]:
                    sum_value += x[w] / degree_out[w]
            x[v] = alpha * sum_value + 1 - alpha
            delta[v] = x[v] - tmp[v]
        if max(delta) < eps:
            break
    norm = np.linalg.norm(x, 1)  # ||x||_1
    x /= norm
    time_end = time.time()
    print("Calculation time:", time_end - time_start, "seconds")
    return x


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: <alpha-value>")
        sys.exit()

    # value of alpha
    Alpha = float(sys.argv[1])

    # reading graph
    Edges, Pages, Node_Index = readGraph()

    # make an adjacent array
    AdjArray = makeAdjArray(Edges)

    # calculating
    PageRank = topologyPageRank(Edges, Pages, Alpha, AdjArray, Node_Index)

    # printing results
    print_highest_lowest_values(PageRank, Pages)
