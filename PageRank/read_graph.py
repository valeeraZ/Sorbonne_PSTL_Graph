import pandas as pd
import time


def readGraph():
    """
    read graph from (i) the list of directed hyperlinks and (ii) the name of the pages corresponding to each node ID
    :return: (list of edges, list of pages, dictionary node_id:node_index)
    """
    time_start = time.time()
    print("Reading files...")

    # Reading
    data_edge = pd.read_table('data/alr21--dirLinks--enwiki-20071018.txt', skiprows=5, dtype=int, header=None)
    data_page = pd.read_table('data/alr21--pageNum2Name--enwiki-20071018.txt', skiprows=5, encoding='utf-8',
                              header=None)
    edges = data_edge.values.tolist()
    pages = data_page.values.tolist()
    print("Number of edges:", len(edges))
    print("Number of nodes:", len(pages))

    # make id-index correspondence
    v = 0
    # dictionary id node:index
    node_to_index = {}
    for node in pages:
        node_to_index[node[0]] = v
        v += 1
    for edge in edges:
        edge[0] = node_to_index[edge[0]]
        edge[1] = node_to_index[edge[1]]
    time_end = time.time()
    print("Charge time:", time_end - time_start, "seconds")
    return edges, pages, node_to_index


def print_highest_lowest_values(page_rank, pages):
    """
    print the 5 highest page rank value with id and name of page and the 5 lowest ones
    :param page_rank: [page_rank_value]
    :param pages: [[id, 'name']]
    :return: nothing
    """
    print("Top 5 highest page rank nodes:")
    t = page_rank.tolist()
    for _ in range(5):
        number = max(t)
        if number >= 0:
            index = t.index(number)
            name = pages[index][1]
            node = pages[index][0]
            print("ID:", node, "Name:", name, "Page Rank value:", number)
            t[index] = -1
    print("Top 5 lowest page rank nodes:")
    t = page_rank.tolist()
    for _ in range(5):
        number = min(t)
        if number <= 1:
            index = t.index(number)
            name = pages[index][1]
            node = pages[index][0]
            print("ID:", node, "Name:", name, "Page Rank value: ", number)
            t[index] = 2


def readInterestedCategories(interested_categories, node_to_index):
    """
    get all pages by interested categories
    :param interested_categories: the interested categories [interested_category_id]
    :param node_to_index: dictionary node_id:node_index {node_id:node_index}
    :return: [pages_index]
    """
    data_page_category = pd.read_table('data/alr21--pageCategList--enwiki--20071018.txt', dtype=str, skiprows=5,
                                       header=None)
    page_categories = data_page_category.values.tolist()
    interested_pages = []
    for page_category in page_categories:
        if isinstance(page_category[1], str):
            categories = page_category[1].split()
            interested = list(set(interested_categories) & set(categories))
            if interested:
                interested_pages.append(page_category[0])
    # turn id to index
    for i in range(len(interested_pages)):
        page = interested_pages[i]
        interested_pages[i] = node_to_index[int(page)]
    return interested_pages


def makeAdjArray(edges):
    """
    make an adjacent array of graph
    :param edges: list of edges [[s,t]]
    :return: adj_array: adjacent array {nodeA: [nodeB, nodeC]}
    """
    adj_array = {}
    for edge in edges:
        if edge[0] not in adj_array.keys():
            adj_array[edge[0]] = [edge[1]]
        else:
            adj_array[edge[0]].append(edge[1])
    return adj_array
