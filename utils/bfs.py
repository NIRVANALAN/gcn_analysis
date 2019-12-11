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
 
print(bfs(graph,'A'))
