from .bfs import bfs_queue, multi_hop_neibs
from typing import List
import numpy as np
import networkx as nx
from tqdm import tqdm

import dgl


def statistics(graph, nodes, level=6):
    from texttable import Texttable
    from tqdm import tqdm
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    results = {}
    stat = [['Node']]
    stat[0].extend([f'{i}-hop' for i in range(1, level+1)])
    for node in tqdm(nodes):
        res, nodes = multi_hop_neibs(graph, node, level)
        row = [node]
        for hop in range(1, level+1):
            if len(res) < hop+1:
                row.append(0)
            else:
                row.append(len(res[hop]))
        stat.append(row)
        results[node] = (res, nodes)
    table.add_rows(stat)
    print(table.draw())
    return results


# degree
def graph_degree(graph: nx.Graph):
    '''
    @param graph: networkx graph
    '''
    if type(graph) is nx.DiGraph:
        graph = graph.to_undirected()
    degree = dict(graph.degree)
    return {
        'max_deg': max(degree.values),
        'avg_deg': np.mean(tuple(degree.values)),
        'min_deg': min(degree.values)
    }


def neighbor_same_label_nxgraph(graph: nx.Graph, train_mask: np.array, labels: np.array):
    """!
    :param graph: networkx graph
    """
    train_nodes = np.nonzero(train_mask)[0]
    same_label_neib_percentage = {}
    for node in tqdm(train_nodes):
        neibs = list(graph.neighbors(node))
        same_label_neib_percentage[node] = (
            (labels[node] == labels[neibs]).mean(), len(neibs))
    return same_label_neib_percentage


def neighbor_same_label_dglgraph(graph: dgl.DGLGraph, train_mask: np.array, labels: np.array):
    train_nodes = np.nonzero(train_mask)[0]
    same_label_neib_percentage = {}
    for node in tqdm(train_nodes):
        neibs = list(graph.successors(node))
        same_label_neib_percentage[node] = (
            (labels[node] == labels[neibs]).mean(), len(neibs))
    return same_label_neib_percentage


def connecvity_of_graph(g: nx.Graph):
    return {
        'is_connected': nx.is_connected(g),
        'number_cc': nx.number_connected_components(g),
        'cc': sorted(nx.connected_components(g), key=len, reverse=True)
    }
