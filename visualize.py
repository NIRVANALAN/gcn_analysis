# %matplotlib inline
import matplotlib.pyplot as plt
import networkx as nx
import randomcolor
import numpy as np


def load_graph(name, root, print_shape=True):
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


def load_label(name, root, mode='raw', print_label=False):
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


def get_colors(label_number=41, binary=True):
    import randomcolor
    rand_color = randomcolor.RandomColor()
    colors = rand_color.generate(count=label_number)
    if binary:
        b_colors = {'true':'green','false':'r'}
    return colors, b_colors

# colors = get_colors(41)

# part_adj, cluster_label = load_graph(773, print_label=True)
# cluster_G = nx.from_scipy_sparse_matrix(part_adj)


def get_node_pos(cluster_label, mode='raw', predict=None):
    node_pos = {}
    if mode == 'raw':
        for label in set(cluster_label):
            node_pos.update(
                {label: np.argwhere(cluster_label == label).flatten().tolist()})
    elif mode == 'diff':
        assert predict is not None
        true = np.argwhere(cluster_label == predict).flatten().tolist()
        false = set(range(len(cluster_label))).difference(true)
        node_pos.update({'true': true, 'false': false})

    else:
        raise NotImplementedError(f'mode: {mode} unrecognized')

    return node_pos


# options = {"node_size": 40, "alpha": 0.8}


def plot_cluster(graph, node_pos, colors, options={"node_size": 40, "alpha": 0.8},  figsize=(8, 6), spring_k=0.15):
    pos = nx.spring_layout(graph, k=spring_k)
    from matplotlib.pyplot import figure
    figure(num=None, figsize=figsize, dpi=150, facecolor='w', edgecolor='k')
    for node_label in node_pos:
        nx.draw_networkx_nodes(
            graph, pos, nodelist=node_pos[node_label], node_color=colors[node_label], **options)
    nx.draw_networkx_edges(graph, pos, width=0.2, alpha=0.5)
