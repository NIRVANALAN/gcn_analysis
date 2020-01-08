import torch
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from typing import List, DefaultDict
from random import sample
from torch.distributions import Categorical

from tqdm import trange, tqdm


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(src, index, num_nodes=None, tau=0, hard=False, eps=1e-20, test=False, monitor: List = None, epoch=0, layer=0, select_topk=-1):
    r"""Computes a sparsely evaluated gumbel softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    # test nodes
    # test_node = 540
    test_node = 10
    test_idx = torch.where(index == test_node)[0].tolist()
    out = softmax(src, index, num_nodes)
    class_prob = out[test_idx]
    if monitor is not None:
        nodes = sample(index.tolist(), 100)
        for node in nodes:
            node_idx = torch.where(index == node)[0].tolist()
            node_out = out[node_idx]
            sorted_probs, _ = torch.sort(node_out, 0, descending=True)
            sorted_list = sorted_probs[:2, :].tolist()
            if len(sorted_list) != 2:  # only one neighbor
                sorted_list.append([0.]*len(sorted_list[0]))
                # import pdb; pdb.set_trace()
            monitor[layer][epoch].append(sorted_list)
            # if epoch > 100 and layer>0:
            #     import pdb; pdb.set_trace()
    if tau > 0:
        # test=True
        out = torch.log(out + eps)
        # out is the categorical distribution
        out += sample_gumbel(out.shape)
        out = softmax(out / tau, index, num_nodes)
        # if test or tau<=0.1:
        if test:
            class_prob_after_tau = out[test_idx]
            print(class_prob)
            print(class_prob_after_tau)
            import pdb
            pdb.set_trace()
    return out


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
