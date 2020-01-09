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


class ThreeLayerResidualGAT(torch.nn.Module):
    def __init__(self, train_dataset, num_features, activation=None):
        super(ThreeLayerResidualGAT, self).__init__()
        self.conv1 = GATConv(num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(
            4 * 256, train_dataset.num_classes, heads=6, concat=False)
        self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)
        self.activation = activation

    def forward(self, x, edge_index, epoch=0, tau=[0, 0, 0], monitor_dict=None):
        x = F.elu(self.conv1(
            x, edge_index, tau[0], epoch=epoch, monitor_dict=monitor_dict, layer=0) + self.lin1(x))
        x = F.elu(self.conv2(
            x, edge_index, tau[1], epoch=epoch, monitor_dict=monitor_dict, layer=1) + self.lin2(x))
        x = self.conv3(x, edge_index, tau[2], epoch=epoch,
                       monitor_dict=monitor_dict, layer=2) + self.lin3(x)
        if self.activation == 'softmax':
            return F.log_softmax(x, 1), self.conv1.alpha
        else:
            return x, self.conv1.alpha
class ThreeLayerGAT(torch.nn.Module):
    def __init__(self, train_dataset, num_features, activation=None):
        super(ThreeLayerGAT, self).__init__()
        self.conv1 = GATConv(num_features, 256, heads=4)
        # self.lin1 = torch.nn.Linear(num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=4)
        # self.lin2 = torch.nn.Linear(4 * 256, 4 * 256)
        self.conv3 = GATConv(
            4 * 256, train_dataset.num_classes, heads=6, concat=False)
        # self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)
        self.activation = activation

    def forward(self, x, edge_index, epoch=0, tau=[0, 0, 0], monitor_dict=None):
        x = F.elu(self.conv1(
            x, edge_index, tau[0], epoch=epoch, monitor_dict=monitor_dict, layer=0))
        x = F.elu(self.conv2(
            x, edge_index, tau[1], epoch=epoch, monitor_dict=monitor_dict, layer=1))
        x = self.conv3(x, edge_index, tau[2], epoch=epoch,
                       monitor_dict=monitor_dict, layer=2)
        if self.activation == 'softmax':
            return F.log_softmax(x, 1), self.conv1.alpha
        else:
            return x, self.conv1.alpha
class TwoLayerResidualGAT(torch.nn.Module):
    def __init__(self, train_dataset, num_features, activation=None):
        super(TwoLayerResidualGAT, self).__init__()
        self.conv1 = GATConv(num_features, 256, heads=4)
        self.lin1 = torch.nn.Linear(num_features, 4 * 256)
        self.conv2 = GATConv(4 * 256, 256, heads=1)
        self.lin2 = torch.nn.Linear(4 * 256, 256)
        # self.conv3 = GATConv(
        #     4 * 256, train_dataset.num_classes, heads=6, concat=False)
        # self.lin3 = torch.nn.Linear(4 * 256, train_dataset.num_classes)
        self.activation = activation

    def forward(self, x, edge_index, epoch=0, tau=[0, 0, 0], monitor_dict=None, return_alpha=True):
        x = F.elu(self.conv1(
            x, edge_index, tau[0], epoch=epoch, monitor_dict=monitor_dict, layer=0) + self.lin1(x))
        x = self.conv2(
            x, edge_index, tau[1], epoch=epoch, monitor_dict=monitor_dict, layer=1) + self.lin2(x)
        # x = self.conv3(x, edge_index, tau[2], epoch=epoch,
        #                monitor_dict=monitor_dict, layer=2) + self.lin3(x)
        ret = []
        if self.activation == 'softmax':
            ret.append(F.log_softmax(x, 1))
        else:
            ret.append(x)
        if return_alpha:
            ret.append(self.conv1.alpha)
        return ret

class TwoLayerBasicGAT(torch.nn.Module):
    def __init__(self, dataset, num_features, activation='softmax', conv2_att_head=8):
        super(TwoLayerBasicGAT, self).__init__()
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(
            8 * 8, dataset.num_classes, heads=conv2_att_head, concat=True, dropout=0.6)
        self.activation = F.log_softmax if activation == 'softmax' else None

    def forward(self, x, edge_index, tau=[0, 0], monitor_dict=None, epoch=0):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index,
                             tau=tau[0], monitor_dict=monitor_dict, epoch=epoch, layer=0))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(
            x, edge_index, tau=tau[1], monitor_dict=monitor_dict, epoch=epoch, layer=1)

        return [self.activation(x, dim=1), self.conv1.alpha]
