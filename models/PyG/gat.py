import os.path as osp
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
# from torch_geometric.nn import GATConv
from nn_local import GATConvGumbel as GATConv
 # from torch_geometric.nn import GATConv

# dataset = 'Cora'
dataset = 'Pubmed'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]

parser = argparse.ArgumentParser()
parser.add_argument('--featureless', action='store_true',
                    help='use identity matric to replace feature x.')
args = parser.parse_args()


num_features = dataset.num_features
if args.featureless:
    # data.x = tor
    data.x = torch.eye(data.x.shape[0])
    num_features = data.x.shape[0]
    pass

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            8 * 8, dataset.num_classes, heads=8, concat=True, dropout=0.6)

    def forward(self, tau=[0,0]):
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, data.edge_index, tau=tau[0]))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, data.edge_index, tau=tau[1])
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

class StepTau():

    def __init__(self, base_taus, step_size=50, gamma=0.1, ori_init=False):
        self.step_size = step_size
        self.gamma = gamma
        self.base_taus = base_taus
        self.ori_init = ori_init
        super(StepTau, self).__init__()

    def get_tau(self, last_epoch):
        if self.ori_init and last_epoch < self.step_size:
            return [0] * len(self.base_taus)
        else:
            return [base_tau * self.gamma ** (last_epoch // self.step_size)
                    for base_tau in self.base_taus]

def train(tau):
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(tau)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).int().sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

best_val, best_test = 0,0
stepTau = StepTau(base_taus=[1,0], step_size=50, gamma=0.5, ori_init=True)
for epoch in range(1, 251):
    tau = stepTau.get_tau(epoch)
    print(tau)
    train(tau)
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    train_loss, val_acc, test_acc = test()
    if best_val < val_acc:
        best_val, best_test = val_acc, test_acc
    print(log.format(epoch, train_loss, val_acc, test_acc))
print(f'Best val_acc: {best_val}, test_acc: {best_test}')
