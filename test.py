from sknetwork.ranking import PageRank
from sknetwork.data import load_edge_list, house

adjacency = house()

pagerank = PageRank(solver='push')
scores = pagerank.fit_transform(adjacency)
print(scores)
"""
the result should be like: [0.17301832 0.22442742 0.1823948  0.18926552 0.23089394]
"""