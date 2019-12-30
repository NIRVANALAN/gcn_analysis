import os.path as osp

import torch
import torch.nn.functional as F
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
import numpy as np
# from torch_geometric.nn import GATConv
from nn_local import GATConvGumbel as GATConv
from sklearn.metrics import f1_score
from utils_local import StepTau
from collections import defaultdict

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PPI')
train_dataset = PPI(path, split='train')
val_dataset = PPI(path, split='val')
test_dataset = PPI(path, split='test')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(train_dataset.num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(train_dataset.num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(
            4 * 256, train_dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)

    def forward(self, x, edge_index, epoch=0, tau=[0, 0, 0], monitor_dict=None):
        x = F.elu(self.conv1(
            x, edge_index, tau[0], epoch=epoch, monitor_dict=monitor_dict, layer=0) + self.lin1(x))
        x = F.elu(self.conv2(
            x, edge_index, tau[1], epoch=epoch, monitor_dict=monitor_dict, layer=1) + self.lin2(x))
        x = self.conv3(x, edge_index, tau[2], epoch=epoch,
                       monitor_dict=monitor_dict, layer=2) + self.lin3(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
loss_op = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train(tau=[0, 0, 0], epoch=0, monitor_dict=None):
    model.train()

    total_loss = 0
    for data in train_loader:
        num_graphs = data.num_graphs
        data.batch = None
        data = data.to(device)
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index,
                             epoch, tau, monitor_dict), data.y)
        total_loss += loss.item() * num_graphs
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader.dataset)


def test(loader):
    model.eval()

    ys, preds = [], []
    for data in loader:
        ys.append(data.y)
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
        preds.append((out > 0).float().cpu())

    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    return f1_score(y, pred, average='micro') if pred.sum() > 0 else 0


stepTau = StepTau(base_taus=[0, 0, 0], step_size=30, gamma=0.3, ori_init=True)
best_val_f1, best_test_f1, epoch_ = 0, 0, 0
monitor_dict = [defaultdict(list), defaultdict(list), defaultdict(list)]
# monitor_dict = None
for epoch in range(1, 101):
    tau = stepTau.get_tau(epoch)
    print(tau)
    loss = train(tau, epoch, monitor_dict)
    val_f1 = test(val_loader)
    test_f1 = test(test_loader)
    if best_val_f1 <= val_f1:
        best_val_f1, best_test_f1, epoch_ = val_f1, test_f1, epoch
    print('Epoch: {:02d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, val_f1, test_f1))
print('Best: Epoch_{:02d}, Val: {:.4f}, Test: {:.4f}'.format(
    epoch_, best_val_f1, best_test_f1))
np.save('ppi_gat_sumtop2', np.array(monitor_dict))