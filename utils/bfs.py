import numpy as np
import torch
import scipy as sp

graph = {'A': ['B', 'C', 'E'],
         'B': ['A','D', 'E'],
         'C': ['A', 'F', 'G'],
         'D': ['B'],
         'E': ['A', 'B','D'],
         'F': ['C'],
         'G': ['C']}
         
         
def bfs(graph, initial):
    
    visited = []
    
    queue = [initial]
 

    while queue:
        
        node = queue.pop(0)
        if node not in visited:
            
            visited.append(node)
            neighbours = graph[node]
 
            
            for neighbour in neighbours:
                queue.append(neighbour)
    return visited
 
# print(bfs(graph,'A'))

def bfs_queue(graph: nx.Graph, root, max_level=3, level=0):
    import collections
    queue, res = collections.deque([(root, 0)]), []
    nodes = set(res)
    while queue:
        node, level = queue.popleft()
        if level > max_level:
            break
        if node not in nodes:
            nodes.update(node)
            if len(res) < level+1:
                res.insert(level, [])
            res[level].append(node)
        for neighbor in list(graph[node]):
            if neighbor not in res:
                queue.append([neighbor, level+1])
    return res, nodes