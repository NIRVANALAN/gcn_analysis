import argparse
import os
import time
import random
import pdb

import numpy as np
import networkx as nx
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_graphs
from dgl.transform import add_self_loop
from torch.utils.tensorboard import SummaryWriter

from modules import GraphSAGE
from sampler import ClusterIter
from utils import Logger, evaluate, save_log_dir, load_data, calc_f1


def main(args):
    torch.manual_seed(args.rnd_seed)
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    multitask_data = set(['ppi'])
    multitask = args.dataset in multitask_data

    # load and preprocess dataset
    assert args.dataset == 'amazon2m'
    g, graph_labels = load_graphs(
        '/yushi/dataset/Amazon2M/Amazon2M_dglgraph.bin')
    assert len(g) == 1
    g = g[0]
    data = g.ndata
    labels = torch.LongTensor(data['label'])
    if hasattr(torch, 'BoolTensor'):
        train_mask = data['train_mask'].bool()
        val_mask = data['val_mask'].bool()
        test_mask = data['test_mask'].bool()

    train_nid = np.nonzero(train_mask.cpu().numpy())[0].astype(np.int64)
    val_nid = np.nonzero(val_mask.cpu().numpy())[0].astype(np.int64)

    # Normalize features
    features = torch.FloatTensor(data['feat'])
    if args.normalize:
        train_feats = features[train_nid]
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(train_feats)
        features = scaler.transform(features)
    features = torch.FloatTensor(features)

    in_feats = features.shape[1]
    n_classes = 47
    n_edges = g.number_of_edges()

    n_train_samples = train_mask.int().sum().item()
    n_val_samples = val_mask.int().sum().item()
    n_test_samples = test_mask.int().sum().item()

    print("""----Data statistics------'
    #Edges %d
    #Classes %d
    #Train samples %d
    #Val samples %d
    #Test samples %d""" %
          (n_edges, n_classes,
           n_train_samples,
           n_val_samples,
           n_test_samples))
    # create GCN model
    if args.self_loop:
        print("adding self-loop edges")
        g = add_self_loop(g)
    # g = DGLGraph(g, readonly=True)

    # set device for dataset tensors
    if args.gpu < 0:
        cuda = False
        raise ValueError('no cuda')
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()

    print(torch.cuda.get_device_name(0))

    g.ndata['features'] = features
    g.ndata['labels'] = labels
    g.ndata['train_mask'] = train_mask
    print('labels shape:', labels.shape)
    train_cluster_iterator = ClusterIter(
        args.dataset, g, args.psize, args.batch_size, train_nid, use_pp=args.use_pp)
    val_cluster_iterator = ClusterIter(
        args.dataset, g, args.psize_val, 1, val_nid, use_pp=False)

    print("features shape, ", features.shape)
    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.use_pp)

    if cuda:
        model.cuda()

    # logger and so on
    log_dir = save_log_dir(args)
    writer = SummaryWriter(log_dir)
    logger = Logger(os.path.join(log_dir, 'loggings'))
    logger.write(args)

    # Loss function
    if multitask:
        print('Using multi-label loss')
        loss_f = nn.BCEWithLogitsLoss()
    else:
        print('Using multi-class loss')
        loss_f = nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)

    # set train_nids to cuda tensor
    if cuda:
        train_nid = torch.from_numpy(train_nid).cuda()
    print("current memory after model before training",
          torch.cuda.memory_allocated(device=train_nid.device) / 1024 / 1024)
    start_time = time.time()
    best_f1 = -1

    for epoch in range(args.n_epochs):
        for j, cluster in enumerate(train_cluster_iterator):
            # sync with upper level training graph
            cluster.copy_from_parent()
            model.train()
            # forward
            pred = model(cluster)
            batch_labels = cluster.ndata['labels']
            batch_train_mask = cluster.ndata['train_mask']
            loss = loss_f(pred[batch_train_mask],
                          batch_labels[batch_train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # in PPI case, `log_every` is chosen to log one time per epoch.
            # Choose your log freq dynamically when you want more info within one epoch
            if j % args.log_every == 0:
                print(f"epoch:{epoch}/{args.n_epochs}, Iteration {j}/"
                      f"{len(train_cluster_iterator)}:training loss", loss.item())
                writer.add_scalar('train/loss', loss.item(),
                                  global_step=j + epoch * len(train_cluster_iterator))
        print("current memory:",
              torch.cuda.memory_allocated(device=pred.device) / 1024 / 1024)

        # evaluate
        if epoch % args.val_every == 0:
            total_f1_mic = []
            total_f1_mac = []
            model.eval()
            for j, cluster in enumerate(val_cluster_iterator):
                cluster.copy_from_parent()
                with torch.no_grad():
                    logits = model(cluster)
                    batch_labels = cluster.ndata['labels']
                    # batch_val_mask = cluster.ndata['val_mask']
                    val_f1_mic, val_f1_mac = calc_f1(batch_labels.cpu().numpy(),
                                                        logits.cpu().numpy(), multitask)
                total_f1_mic.append(val_f1_mic)
                total_f1_mac.append(val_f1_mac)

            val_f1_mic = np.mean(total_f1_mic)
            val_f1_mac = np.mean(total_f1_mac)

            print(
                "Val F1-mic{:.4f}, Val F1-mac{:.4f}". format(val_f1_mic, val_f1_mac))
            if val_f1_mic > best_f1:
                best_f1 = val_f1_mic
                print('new best val f1:', best_f1)
                torch.save(model.state_dict(), os.path.join(
                    log_dir, 'best_model.pkl'))
            writer.add_scalar('val/f1-mic', val_f1_mic, global_step=epoch)
            writer.add_scalar('val/f1-mac', val_f1_mac, global_step=epoch)

    end_time = time.time()
    print(f'training using time {start_time-end_time}')

    # test
    if args.use_val:
        model.load_state_dict(torch.load(os.path.join(
            log_dir, 'best_model.pkl')))
    # test_f1_mic, test_f1_mac = evaluate(
    #     model, g, labels, test_mask, multitask)
    # print(
    #     "Test F1-mic{:.4f}, Test F1-mac{:.4f}". format(test_f1_mic, test_f1_mac))
    # writer.add_scalar('test/f1-mic', test_f1_mic)
    # writer.add_scalar('test/f1-mac', test_f1_mac)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=3e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--log-every", type=int, default=100,
                        help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="batch size")
    parser.add_argument("--psize", type=int, default=15000,
                        help="partition number")
    parser.add_argument("--psize-val", type=int, default=200,
                        help="partition number val")
    parser.add_argument("--test-batch-size", type=int, default=1000,
                        help="test batch size")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--val-every", type=int, default=1,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--rnd-seed", type=int, default=3,
                        help="number of epoch of doing inference on validation")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--use-pp", action='store_true',
                        help="whether to use percomputation")
    parser.add_argument("--normalize", action='store_true',
                        help="whether to use normalized feature")
    parser.add_argument("--use-val", action='store_true',
                        help="whether to use validated best model to test")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--note", type=str, default='none',
                        help="note for log dir")

    args = parser.parse_args()

    print(args)

    main(args)
