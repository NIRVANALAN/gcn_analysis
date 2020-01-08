import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T

import numpy as np
# from torch_geometric.nn import GATConv
from nn_local import GATConvGumbel as GATConv
from utils_local import StepTau, ClassBoundaryLoss
# from torch_geometric.nn import GATConv
from collections import defaultdict

from models import ThreeLayerResidualGAT, TwoLayerResidualGAT, TwoLayerBasicGAT

def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


dataset = 'Cora'
# dataset = 'Pubmed'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]
cora_train_index = torch.arange(1208, dtype=torch.long)
cora_val_index = torch.arange(cora_train_index[-1]+1, cora_train_index[-1]+1 + 500, dtype=torch.long)
data.train_mask = index_to_mask(cora_train_index, size=data.y.shape[0])
data.val_mask = index_to_mask(cora_val_index, size=data.y.shape[0])


parser = argparse.ArgumentParser()
parser.add_argument('--featureless', action='store_true',
                    help='use identity matric to replace feature x.')
args = parser.parse_args()


num_features = dataset.num_features
if args.featureless:
    # data.x = tor
    data.x = torch.eye(data.x.shape[0])
    num_features = data.x.shape[0]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = TwoLayerBasicGAT(
    dataset, num_features, conv2_att_head=8).to(device), data.to(device)
# model, data = TwoLayerResidualGAT(
#     dataset, num_features, activation='softmax').to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

margin_loss_settings = {'factor': 1, 'margin': 0.3}
class_boundary_loss = ClassBoundaryLoss(
    margin=margin_loss_settings['margin'])


def train(tau, monitor_dict, epoch):
    model.train()
    loss = 0.
    optimizer.zero_grad()
    ret_val = model(data.x, data.edge_index, tau=tau,
                    monitor_dict=monitor_dict, epoch=epoch)
    preds, conv1_alpha = ret_val[0][data.train_mask], ret_val[1]
    cls_loss = F.nll_loss(preds, data.y[data.train_mask])  # nllloss
    cb_loss = class_boundary_loss(
        conv1_alpha, data.y[:], data.edge_index, data.train_mask, nodes=data.x.shape[0]) * margin_loss_settings['factor']
    # loss = cls_loss
    loss = cls_loss + cb_loss 
    loss.backward()
    optimizer.step()
    return cls_loss, cb_loss, 0


def test():
    model.eval()
    logits, accs = model(data.x, data.edge_index)[0], []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).int().sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


best_epoch, best_val, best_test = 0, 0, 0
# stepTau = StepTau(base_taus=[0, 0], step_size=60, gamma=0.5, ori_init=True)
stepTau = StepTau(base_taus=[0, 0, 0], step_size=60, gamma=0.5, ori_init=True)
# monitor_dict = [defaultdict(list), defaultdict(list)]
monitor_dict = [defaultdict(list), defaultdict(list), defaultdict(list)]
for epoch in range(1, 201):
    tau = stepTau.get_tau(epoch)
    print(tau)
    cls_loss, cb_loss, gs_loss = train(tau, monitor_dict, epoch)
    log = 'Epoch: {:03d}, Cls_loss: {:.4f}, Cb_loss:{:.4f}, Gs_loss:{:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    train_acc, val_acc, test_acc = test()
    if best_val < val_acc:
        best_epoch, best_val, best_test = epoch, val_acc, test_acc,
    print(log.format(epoch, cls_loss, cb_loss,
                     gs_loss, train_acc, val_acc, test_acc))
print(
    f'Best Epoch: {best_epoch}, Best val_acc: {best_val}, test_acc: {best_test}')

np.save('cora_gat_sumtop2', np.array(monitor_dict))
