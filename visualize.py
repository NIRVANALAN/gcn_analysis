# %matplotlib inline
from typing import Dict, Tuple, List, Sequence, Optional
import matplotlib.pyplot as plt
import networkx as nx
import randomcolor
import numpy as np
import torch

'''
TODO add thicker edge for error case
TODO add graph construction function with error case as center
'''

def load_graph_from_npz(name, root, print_shape=True):
    import scipy.sparse as sp
    import networkx as nx
    from pathlib import PurePath
    path = PurePath(root, name)
    if path.suffix is not '.npz':
        path = path.with_suffix('.npz')
    part_adj = sp.load_npz(path)
    graph = nx.from_scipy_sparse_matrix(part_adj)
    if print_shape:
        print(f'shape: {part_adj.shape}')
    return part_adj, graph


def load_label_from_npy(name, root, mode='raw', print_label=False):
    import numpy as np
    from pathlib import PurePath
    path = PurePath(root, name)
    if path.suffix is not '.npy':
        path = path.with_suffix('.npy')
    graph_label = np.load(path)

    if print_label:
        print_label_table(graph_label)
    if graph_label.shape == 2:
        graph_label = np.argmax(graph_label, axis=1)
    node_pos = get_node_pos(graph_label, mode)
    return graph_label, node_pos


def print_label_table(label):
    from tabletext import to_text
    from collections import Counter
    labeled_node = np.argwhere(label)[:, 1] if label.shape == 2 else label
    print(f'all_node: {len(label)}, labeld: {len(labeled_node)}')
    label_stat = Counter(labeled_node)
    label_table = [['label', 'number', 'percent']]
    for _label_tuple in label_stat.most_common():
        label_table.append([_label_tuple[0], _label_tuple[1],
                            f'{_label_tuple[1]/len(labeled_node)*100:.2f}%'])
    print(to_text(label_table))
    pass


def get_colors(label_number: int, mode: str):
    import randomcolor
    rand_color = randomcolor.RandomColor()
    if mode == 'diff':
        colors = {'train': 'grey', 'pred_train': 'green', 'pred_false': 'r'}
    elif mode == 'raw':
        colors = rand_color.generate(count=label_number)
    elif mode == 'train_test':
        colors = {'train': 'grey', 'test': 'black'}
    else:
        raise NotImplementedError
    return colors

# colors = get_colors(41)

# part_adj, cluster_label = load_graph(773, print_label=True)
# cluster_G = nx.from_scipy_sparse_matrix(part_adj)


def get_node_pos(labels: Optional[np.ndarray] = None, sg_nodes: List = None, idx_test: Optional[List] = None, mode='raw', predict: Optional[np.ndarray] = None) -> Dict:
    '''
    1. get the whole graph: raw
    2. train and test node
    3. right and wrong node
    '''
    if type(labels) is torch.Tensor:
        labels = labels.cpu().numpy()
    if type(predict) is torch.Tensor:
        predict = predict.cpu().numpy()

    node_pos = {}
    # np.argwhere(labels == label).flatten().tolist()})
    if mode == 'raw':
        if sg_nodes is not None:
            for label in set(labels[sg_nodes]):
                node_pos.update(
                    {label: [node for node in sg_nodes if labels[node] == label]})
        else:
            for label in set(labels):
                node_pos.update(
                    {label: np.argwhere(labels == label).flatten().tolist()})

    elif mode == 'train_test':
        assert idx_test is not None and sg_nodes is not None
        test_sg_nodes = list(set(sg_nodes).intersection(idx_test))
        train_sg_nodes = list(set(sg_nodes).difference(test_sg_nodes))
        node_pos.update({'train': train_sg_nodes, 'test': test_sg_nodes})
        pass

    elif mode == 'diff':
        assert predict is not None and sg_nodes is not None and idx_test is not None
        test_sg_nodes = list(set(sg_nodes).intersection(idx_test))
        train_sg_nodes = list(set(sg_nodes).difference(test_sg_nodes))
        true = list(set(idx_test[np.argwhere(
            labels[idx_test] == predict).flatten().tolist()]).intersection(test_sg_nodes))
        false = list(set(test_sg_nodes).difference(true))
        node_pos.update(
            {'train': train_sg_nodes, 'pred_true': true, 'pred_false': false})

    else:
        raise NotImplementedError(f'mode: {mode} unrecognized')

    return node_pos

# options = {"node_size": 40, "alpha": 0.8}


# pos = nx.spring_layout(graph, k=spring_k)
def plot_cluster(graph, node_pos, colors, pos, options={"node_size": 40, "alpha": 0.8},  figsize=(8, 6), spring_k=0.15):
    from matplotlib.pyplot import figure
    figure(num=None, figsize=figsize, dpi=150, facecolor='w', edgecolor='k')
    for node_label in node_pos:
        nx.draw_networkx_nodes(
            graph, pos, nodelist=node_pos[node_label], node_color=colors[node_label], **options)
    nx.draw_networkx_edges(graph, pos, width=0.2, alpha=0.5)
