# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collections of partitioning functions."""

import time
import metis
import scipy.sparse as sp
import tensorflow as tf
import numpy as np
from collections import Counter
from tabletext import to_text 
import networkx as nx
import pdb


def partition_graph_by_label(label):
    groups = []
    # import pdb; pdb.set_trace()
    for node in range(label.shape[0]):
        node_label = np.argwhere(label[node, :]).squeeze(-1)
        node_label = node_label[0] if len(node_label) else -1
        groups.append(node_label)
    return groups

def print_label_table(label):
  from tabletext import to_text 
  from collections import Counter
  labeled_node = np.argwhere(label)[:,1]
  print(f'all_node: {len(label)}, labeld: {len(labeled_node)}')
  label_stat = Counter(labeled_node)
  label_table = [['label', 'number','percent']]
  for _label_tuple in label_stat.most_common():
    label_table.append([_label_tuple[0], _label_tuple[1], f'{_label_tuple[1]/len(labeled_node)*100:.2f}%'])
  print(to_text(label_table))
  pass

def partition_graph(adj, idx_nodes, num_clusters, label=None, lable_cluster=False, stat=False, visu=False):
    """
    partition a graph by METIS.
    idx_nodes: visible_data (train_data)
    y: add lable constraint
    """
    if lable_cluster:
      assert label is not None

    start_time = time.time()
    num_nodes = len(idx_nodes)
    num_all_nodes = adj.shape[0]
    neighbor_intervals = []
    neighbors = []
    edge_cnt = 0
    neighbor_intervals.append(0)
    train_adj_lil = adj[idx_nodes, :][:, idx_nodes].tolil()
    train_ord_map = dict()
    train_adj_lists = [[] for _ in range(num_nodes)]
    for i in range(num_nodes):
        rows = train_adj_lil[i].rows[0]  # columns of each row
        # self-edge needs to be removed for valid format of METIS
        if i in rows:
            rows.remove(i)  # remove self edges
        train_adj_lists[i] = rows
        neighbors += rows
        edge_cnt += len(rows)
        neighbor_intervals.append(edge_cnt)
        train_ord_map[idx_nodes[i]] = i

    if lable_cluster:
        groups = partition_graph_by_label(label)
        # set cluster number to the size of label
        num_clusters = label.shape[1]
        print(f'num_clusters: {num_clusters}, partition graph by label')
    else:
        print(f'start clustering, num_clusters:{num_clusters}')
        if num_clusters > 1:
            _, groups = metis.part_graph(train_adj_lists, num_clusters, seed=1)
        else:
            groups = [0] * num_nodes  # TODO,cluster based on labels

    part_row = []
    part_col = []
    part_data = []
    parts = [[] for _ in range(num_clusters)]
    for nd_idx in range(num_nodes):
        gp_idx = groups[nd_idx]
        if gp_idx < 0:
            continue
        nd_orig_idx = idx_nodes[nd_idx]
        parts[gp_idx].append(nd_orig_idx)
        # neibourhood index of node nd_orig_idx
        for nb_orig_idx in adj[nd_orig_idx].indices:
            nb_idx = train_ord_map[nb_orig_idx]
            if groups[nb_idx] == gp_idx:
                part_data.append(1)
                part_row.append(nd_orig_idx)
                part_col.append(nb_orig_idx)
    # statistics -----
    if stat and label is not None:
        print_label_table(label)
        for cluster in parts:
            cluster_label = label[cluster, :]
            print_label_table(cluster_label)
            import pdb; pdb.set_trace()

    # ---------------
    part_data.append(0)
    part_row.append(num_all_nodes - 1)
    part_col.append(num_all_nodes - 1)  # guarantee boundary of coo_matrix
    part_adj = sp.coo_matrix((part_data, (part_row, part_col))).tocsr()
    # sp.save_npz('cluster.npz', part_adj)

    # if visu: # visualize the cluster
    #   cluster_G = nx.from_scipy_sparse_matrix(part_adj)
    #   nx.draw(cluster_G, pos=nx.spring_layout(cluster_G))
    #   pass

    tf.logging.info('Partitioning done. %f seconds.', time.time() - start_time)
    return part_adj, parts
