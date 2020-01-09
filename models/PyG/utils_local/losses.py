import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from tqdm import tqdm, trange
from scipy import sparse


class ClassBoundaryLoss(_Loss):
    __constants__ = ['reduction', 'flow', 'edge_index', 'margin']

    def __init__(self, margin, size_average=None, reduce=None, reduction='mean', flow='source_to_target'):  # by default is fine
        super(ClassBoundaryLoss, self).__init__(
            size_average, reduce, reduction)
        assert flow in ['source_to_target', 'target_to_source']
        self.i, self.j = (0, 1) if flow == 'target_to_source' else (1, 0)
        self.margin = margin
        self.flag = 150
        # self.sparse_lil_matrix = None

    def forward(self, attention, class_target, edge_index, idx_mask, nodes):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=nodes)
        class_boundary_loss = torch.tensor(0.).to(attention.device)
        # create til format sparse adj matrix
        # import pdb; pdb.set_trace() 
        # data = np.ones(len(edge_index[0]))
        # sparse_lil_matrix = sparse.lil_matrix(sparse.coo_matrix(data, edge_index.tolist()))

        self.idx_mask = torch.nonzero(idx_mask).flatten()
        # self.flag -= 1
        for idx in range(len(self.idx_mask)):
            node_idx = self.idx_mask[idx]
            neibs_coo = torch.where(edge_index[self.i] == node_idx)[0]
            pos_neibs, neg_neibs = [], []
            for neib in (neibs_coo.flatten()):
                # if class_target[node_idx].shape[0] > 1:
                #     if all(class_target[edge_index[self.j][neib]] == class_target[node_idx]):
                #         pos_neibs.append(attention[neib])
                if class_target[edge_index[self.j][neib]] == class_target[node_idx]:
                    pos_neibs.append(attention[neib])
                else:
                    neg_neibs.append(attention[neib])
            if pos_neibs and neg_neibs:
                pos_neibs, neg_neibs = torch.stack(pos_neibs).sum(
                    0), torch.stack(neg_neibs).sum(0)
                # import pdb; pdb.set_trace()
                # class_boundary_loss += torch.relu(
                #     neg_neibs - pos_neibs + self.margin).mean()
                for pos_neib_att in pos_neibs:
                    for neg_neib_att in neg_neibs:
                        if self.flag <=0:
                            import pdb; pdb.set_trace()
                        class_boundary_loss += torch.relu(
                            neg_neib_att - pos_neib_att + self.margin).mean() # 1.head 2.
        # import pdb; pdb.set_trace()
            # same_class_flag
        if self.reduction == 'mean':
            class_boundary_loss /= len(self.idx_mask)
        return class_boundary_loss


# class GraphStructureLoss(_Loss):
#     __constants__ = ['reduction', 'flow', 'edge_index', 'margin']

#     def __init__(self, edge_index, idx_mask, margin, size_average=None, reduce=None, reduction='mean', flow='source_to_target'):  # by default is fine
#         super(GraphStructureLoss, self).__init__(
#             size_average, reduce, reduction)
#         assert flow in ['source_to_target', 'target_to_source']
#         self.i, self.j = (0, 1) if flow == 'target_to_source' else (1, 0)
#         self.idx_mask = torch.nonzero(idx_mask).flatten()
#         self.margin = margin
#         self.edge_index = edge_index

#     def forward(self, attention, class_target):
#         class_boundary_loss = 0.
#         for idx in range(len(self.idx_mask)):
#             node_idx = self.idx_mask[idx]
#             neibs_coo = torch.where(self.edge_index[self.i] == node_idx)[0]
#             pos_neibs, neg_neibs = [], []
#             for neib in neibs_coo.flatten():
#                 if class_target[self.edge_index[self.j][neib]] == class_target[node_idx]:
#                     pos_neibs.append(attention[neib])
#                 else:
#                     neg_neibs.append(attention[neib])
#             if pos_neibs and neg_neibs:
#                 # pos_neibs, neg_neibs = torch.LongTensor(pos_neibs).to(attention.device), torch.LongTensor(neg_neibs).to(attention.device)
#                 for pos_neib_att in pos_neibs:
#                     for neg_neib_att in neg_neibs:
#                         class_boundary_loss += torch.relu(
#                             neg_neib_att - pos_neib_att + self.margin).mean()
#         if self.reduction == 'mean':
#             class_boundary_loss /= len(self.idx_mask)
#         import pdb; pdb.set_trace()
#         return class_boundary_loss
