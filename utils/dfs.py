# %%
from collections import defaultdict
import numpy as np
import torch
import scipy as sp
import networkx as nx
import collections
from typing import List, Tuple, Dict

# Using a Python dictionary to act as an adjacency list
graph = nx.Graph({
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F'],
    'D': ['G'],
    'E': ['F', 'Z'],
    'Z': ['T','H'],
    # 'H': [1,2,3,4,5],
    'F': [],
    'G': ['C']
})

# graph = graph.to_undirected()

print(f'graph: {list(graph.edges)}')
# visited = [] # Array to keep track of visited nodes.
# %%
traverse = defaultdict(int)


def dfs_recursively(graph: nx.Graph, root, visited, level=0):
    if root not in visited or visited[root] > level:
        visited[root] = level
    for neighbor in list(graph[root]):
        if neighbor not in visited:
            dfs_recursively(graph, neighbor, visited, level+1)
        if visited[neighbor] > level+1:
            visited[neighbor] = level + 1
    return visited


def bfs_queue(graph: nx.Graph, root, level=0, max_level=3):
    import collections
    queue, res = collections.deque([(root, 0)]), {}
    while queue:
        node, level = queue.popleft()
        if level > max_level:
            break
        if node not in res: 
            res[node] = level
        for neighbor in list(graph[node]):
            if neighbor not in res:
                queue.append([neighbor, level+1])
    return res


# # Driver Code
print(bfs_queue(graph, 'A'))
