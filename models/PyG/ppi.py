import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
import numpy as np
# from torch_geometric.nn import GATConv
from nn_local import GATConvGumbel as GATConv
from utils_local import StepTau, ClassBoundaryLoss
from sklearn.metrics import f1_score
from collections import defaultdict

from models import ThreeLayerResidualGAT

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ThreeLayerResidualGAT(
    train_dataset, train_dataset.num_features).to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

margin_loss_settings = {'factor': 10, 'margin': 0.5}
class_boundary_loss = ClassBoundaryLoss(
    margin=margin_loss_settings['margin'], )


def train(tau=[0, 0, 0], epoch=0, monitor_dict=None):
    model.train()

    total_loss = 0
    cb_total_loss = 0.
    for data in train_loader:
        # import pdb; pdb.set_trace()
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        ret_val = model(data.x, data.edge_index, epoch, tau, monitor_dict)
        preds, conv1_alpha = ret_val[0], ret_val[1]
        cls_loss = loss_op(preds, data.y)
        cb_loss = class_boundary_loss(
            conv1_alpha, data.y[:], data.edge_index, torch.arange(data.x.shape[0]), nodes=data.x.shape[0]) * margin_loss_settings['factor']
        total_loss += cls_loss.item() * num_graphs 
        cb_total_loss += cb_loss.item() * num_graphs
        (cls_loss+cb_loss).backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset), cb_total_loss / len(train_loader.dataset)


def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out[0] > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


stepTau = StepTau(base_taus=[0, 0, 0], step_size=30, gamma=0.3, ori_init=True)
best_val_f1, best_test_f1, epoch_ = 0, 0, 0
monitor_dict = [defaultdict(list), defaultdict(list), defaultdict(list)]
# monitor_dict = None
for epoch in range(1, 101):
    tau = stepTau.get_tau(epoch)
    print(tau)
    cls_loss, cb_loss = train(tau, epoch, monitor_dict)
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    if best_val_f1 <= val_f1:
        best_val_f1, best_test_f1, epoch_ = val_f1, test_f1, epoch
    print('Epoch: {:02d}, ClsLoss: {:.4f}, CBLoss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
        epoch, cls_loss, cb_loss, val_f1, test_f1))
print('Best: Epoch_{:02d}, Val: {:.4f}, Test: {:.4f}'.format(
    epoch_, best_val_f1, best_test_f1))
np.save('ppi_gat_sumtop2', np.array(monitor_dict))
