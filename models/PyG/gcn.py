import math
import sys
import os
import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv  # noqa

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
parser.add_argument('--featureless', action='store_true',
                    help='use identity matric to replace feature x.')
args = parser.parse_args()

dataset = 'Cora'
# dataset = 'Pubmed'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

if args.use_gdc:
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)
num_features = dataset.num_features
if args.featureless:
    # data.x = tor
    data.x = torch.eye(data.x.shape[0])
    num_features = data.x.shape[0]
    pass
def construct_edge_mask(num_nodes, init_strategy="const", const_val=1.0, mask_bias=False):
    mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
    if init_strategy == "normal":
        std = nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (num_nodes + num_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
            # mask.clamp_(0.0, 1.0)
    elif init_strategy == "const":
        nn.init.constant_(mask, const_val)

    if mask_bias:
        mask_bias = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        nn.init.constant_(mask_bias, 0.0)
    else:
        mask_bias = None

    return mask, mask_bias

def construct_feat_mask(num_nodes, feat_dim, init_strategy="normal"):
    mask = nn.Parameter(torch.FloatTensor(num_nodes, feat_dim))
    if init_strategy == "normal":
        std = 0.1
        with torch.no_grad():
            mask.normal_(1.0, std)
    elif init_strategy == "constant":
        with torch.no_grad():
            nn.init.constant_(mask, 0.0)
            # mask[0] = 2
    return mask

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(num_features, 16, cached=True,
                            #  normalize=not args.use_gdc
                             )
        self.conv2 = GCNConv(16, dataset.num_classes, cached=True,
                            #  normalize=not args.use_gdc
                             )
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)
        self.mask, self.mask_bias = construct_edge_mask(
            data.x.shape[0],init_strategy='const')
        self.feat_mask = construct_feat_mask(data.x.shape[0], num_features, init_strategy="normal")
        # mask, feat_mask = mask.to(device), feat_mask.to(device)
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self, x, edge_index, edge_weight ):
        # x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        self.feat_mask =nn.Parameter(torch.sigmoid(self.feat_mask))
        x *= self.feat_mask
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1), self.feat_mask


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam([
    dict(params=model.reg_params, weight_decay=5e-4),
    dict(params=model.non_reg_params, weight_decay=0),
    dict(params=model.mask, weight_decay=0),
    dict(params=model.feat_mask, weight_decay=0),
], lr=0.01)

coeffs = {
        "size": 0.005,
        "feat_size": 1.0,
        "ent": 1.0,
        "feat_ent": 0.1,
        "grad": 0,
        "lap": 1.0,
    }


# construct mask
# data.x.shape[0], num_features


def _masked_adj(mask):
    sym_mask = mask
    if self.mask_act == "sigmoid":
        sym_mask = torch.sigmoid(self.mask)
    elif self.mask_act == "ReLU":
        sym_mask = nn.ReLU()(self.mask)
    sym_mask = (sym_mask + sym_mask.t()) / 2
    adj = self.adj.cuda() if self.args.gpu else self.adj
    masked_adj = adj * sym_mask
    if self.args.mask_bias:
        bias = (self.mask_bias + self.mask_bias.t()) / 2
        bias = nn.ReLU6()(bias * 6) / 6
        masked_adj += (bias + bias.t()) / 2
    return masked_adj * self.diag_mask

# class HLoss(nn.Module):
#     def __init__(self):
#         super(HLoss, self).__init__()

#     def forward(self, x):
#         b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
#         b = -1.0 * b.sum()
#         return b

def train():
    model.train()
    optimizer.zero_grad()
    # losses
    # feat_mask = (
    #             torch.sigmoid(self.feat_mask)
    #             if self.use_sigmoid else self.feat_mask)
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
    # import pdb; pdb.set_trace()
    #train
    logits, feat_mask = model(x, edge_index, edge_weight)
    pred_loss = F.nll_loss(logits[data.train_mask], data.y[data.train_mask])

    # size_loss = coeffs['size'] * torch.sum(mask)
    # mask_ent = -mask * torch.log(mask) - (1 - mask) * torch.log(1 - mask)
    # mask_ent_loss = coeffs["ent"] * torch.mean(mask_ent)
    # mask_ent_loss
    feat_mask_ent = - feat_mask * torch.log(feat_mask) - (1 - feat_mask)* torch.log(1 - feat_mask)
    feat_mask_ent_loss = coeffs["feat_ent"] * torch.mean(feat_mask_ent)
    feat_size_loss = coeffs["feat_size"] * torch.mean(feat_mask)
    loss = pred_loss + feat_size_loss + feat_mask_ent_loss
    # import pdb; pdb.set_trace()
    #loss = pred_loss + mask_ent_loss + size_loss + feat_size_loss
    # update param
    loss.backward(retain_graph=True)
    optimizer.step()


def test():
    model.eval()
    x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
    logits, accs = model(x, edge_index, edge_weight)[0], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.int().sum().item()
        accs.append(acc)
    return accs


best_val_acc = test_acc = 0
for epoch in range(1, 101):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, train_acc, best_val_acc, test_acc))
