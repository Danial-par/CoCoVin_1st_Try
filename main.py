import argparse
import torch
import sys
import os
import models
import models_ogb
import trainers
import torch_geometric as pyg
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric import transforms
import numpy as np
import time
import pandas as pd

def save_results(history, filename):
    """
    Saves the training, validation, and test accuracy and loss over epochs to a CSV file.
    """
    df = pd.DataFrame(history)
    df.index.name = 'epoch'
    df.to_csv(filename)
    print(f"Results saved to {filename}")

def ogb_prep(db, db_dir):
    dataset = PygNodePropPredDataset(name=db, root=db_dir)
    split_idx = dataset.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    # the original graph is a directional graph, we should convert it into a bi-directional graph
    graph = dataset[0]
    label = graph.y

    # add reverse edges
    edge_index = graph.edge_index
    print(f"Total edges before adding reverse edges: {len(edge_index[0])}")
    reverse_edge_index = torch.cat([edge_index[1].unsqueeze(0), edge_index[0].unsqueeze(0)], dim=0)
    edge_index = torch.cat((edge_index, reverse_edge_index), dim=1)
    # remove duplicate edges
    edge_index = pyg.utils.coalesce(edge_index)
    graph.edge_index = edge_index
    print(f"Total edges after adding reverse edges: {len(edge_index[0])}")

    train_mask = torch.zeros(label.shape[0]).scatter_(0, train_idx, torch.ones(train_idx.shape[0])).bool()
    valid_mask = torch.zeros(label.shape[0]).scatter_(0, valid_idx, torch.ones(valid_idx.shape[0])).bool()
    test_mask = torch.zeros(label.shape[0]).scatter_(0, test_idx, torch.ones(test_idx.shape[0])).bool()
    graph.train_mask = train_mask
    graph.val_mask = valid_mask
    graph.test_mask = test_mask
    graph.y = label.squeeze()

    return graph, dataset.num_classes

def create_standard_split(data, num_per_class):
    # Get the number of nodes and classes
    num_nodes = data.x.size(0)
    num_classes = data.y.max().item() + 1

    # Initialize masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # For each class, sample num_per_class nodes for training
    # 30% of remaining nodes for validation, 70% for testing
    for c in range(num_classes):
        # Find nodes of this class
        idx = (data.y == c).nonzero(as_tuple=True)[0]

        # Shuffle the indices
        idx = idx[torch.randperm(idx.size(0))]

        # Select training nodes (20 per class)
        train_nodes = idx[:num_per_class]
        train_mask[train_nodes] = True

        # Remaining nodes
        remaining = idx[num_per_class:]
        n_val = int(remaining.size(0) * 0.3)

        # Split between validation and test
        val_nodes = remaining[:n_val]
        test_nodes = remaining[n_val:]
        val_mask[val_nodes] = True
        test_mask[test_nodes] = True

    # Add the masks to the data object
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

def load_data(db, db_dir='./dataset'):
        if db in ['Cora', 'CiteSeer', 'PubMed']:
            data = Planetoid(db_dir, db, transform=transforms.NormalizeFeatures())
            g = data[0]
            out_dim = data.num_classes
        elif db in ['photo', 'computers']:
            data = Amazon(db_dir, db, transform=transforms.NormalizeFeatures())
            g = data[0]
            out_dim = data.num_classes
            # Create standard split with 20 labels per class
            g = create_standard_split(g, 20)
        elif db in ['cs', 'physics']:
            data = Coauthor(db_dir, db, transform=transforms.NormalizeFeatures())
            g = data[0]
            out_dim = data.num_classes
            # Create standard split with 20 labels per class
            g = create_standard_split(g, 20)
        elif db == 'ogbn-arxiv':
            g, out_dim = ogb_prep('ogbn-arxiv', db_dir)
        elif db == 'ogbn-products':
            g, out_dim = ogb_prep('ogbn-products', db_dir)
        else:
            raise ValueError('Unknown dataset: {}'.format(db))

        info_dict = {
            'in_dim': g.x.shape[1],
            'out_dim': out_dim
        }
        return g, info_dict

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    return

def main(args):
    acc_list = []
    time_cost_list = []

    # Create a directory for results if it doesn't exist
    results_dir = os.path.join('exp', 'results')
    os.makedirs(results_dir, exist_ok=True)

    for i in range(args.round):
        g, info_dict = load_data(args.dataset)
        set_seed(args.seed)
        info_dict.update(args.__dict__)
        info_dict.update({'device': torch.device('cpu') if args.gpu == -1 else torch.device('cuda:{}'.format(args.gpu)),})

        args.seed = args.seed + 1

        # Define backbone models that have OGB versions
        backbone_list = ['GCN', 'GAT', 'SAGE', 'JKNet', 'GCN2', 'APPNPNet', 'GIN', 'SGC']

        # Initialize model
        model = getattr(models_ogb if args.dataset == 'ogbn-arxiv' else models, args.model)(info_dict)

        # Initialize trainer
        if args.model in backbone_list:
            trainer = getattr(trainers, 'BaseTrainer')(g, model, info_dict)
        elif args.model.startswith('CoCoVin'):
            info_dict.update({'backbone': args.model[7:]})
            Dis = getattr(models_ogb if args.dataset == 'ogbn-arxiv' else models, 'DisMLP')(info_dict)
            Dis.to(info_dict['device'])
            trainer = getattr(trainers, 'CoCoVinTrainer')(g, model, info_dict, Dis=Dis)
        elif args.model.startswith('Violin'):
            info_dict.update({'backbone': args.model[6:]})
            trainer = getattr(trainers, 'ViolinTrainer')(g, model, info_dict)
        else:
            raise ValueError('Unknown model: {}'.format(args.model))

        model.to(info_dict['device'])
        print(model)
        print('\nSTART TRAINING\n')
        tic = time.time()
        val_acc, tt_acc, val_acc_fin, tt_acc_fin, microf1, macrof1 = trainer.train()
        toc = time.time()

        acc_list.append(tt_acc)
        time_cost = toc - tic
        time_cost_list.append(time_cost)
        print('The time cost of the {} round ({} epochs) is: {}.'.format(i, info_dict['n_epochs'], time_cost))

    print('\n\n')
    print('The averaged accuracy of {} rounds of experiments on {} is: {}'.format(args.round, args.dataset, np.mean(acc_list)))
    print('The averaged time cost (seconds/ 100 epochs) of {} rounds is {:.4f}'.format(args.round, np.mean(time_cost_list) / args.n_epochs * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='the main program to run experiments on small datasets')
    parser.add_argument("--round", type=int, default=10,
                        help="number of rounds to repeat the experiment")
    parser.add_argument("--model", type=str, default='ViolinGCN',
                        help="model name")
    parser.add_argument("--dataset", type=str, default='Cora',
                        help="the dataset for the experiment (Cora, CiteSeer, PubMed, photo, computer, cs, physics, ogbn-arxiv, ogbn-products)")
    parser.add_argument("--n_epochs", type=int, default=300,
                        help="the number of training epochs")
    parser.add_argument("--eta", type=int, default=1,
                        help="the interval (epoch) to override/ update the estimated labels")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="the number of hidden layers")
    parser.add_argument("--hid_dim", type=int, default=16,
                        help="the hidden dimension of hidden layers in the backbone model")
    parser.add_argument("--dropout", type=float, default=0.6,
                        help="dropout rate")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="specify the gpu index, set -1 to train on cpu")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="the learning rate")
    parser.add_argument("--lr_cocos", type=float, default=0.01,
                        help="the learning rate for the CoCoS optimizer, only for CoCoS-enhanced models")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="the weight decay for optimizer")
    parser.add_argument("--alpha", type=float, default=0.8,
                        help="coefficient for the consistency loss")
    parser.add_argument("--gamma", type=float, default=0.6,
                        help="coefficient for the VO loss")
    parser.add_argument("--cls_mode", type=str, default='virt',
                        help="the type of the classification loss (Eq.10), 'virt' only includes the second term, while 'both' inlcudes both terms")
    parser.add_argument("--delta", type=float, default=0.9,
                        help="the acc requirement (\delta) to pick node candidates for building VOs")
    parser.add_argument("--m", type=int, default=1,
                        help="the number of VOs per node")
    parser.add_argument("--bn", action='store_true', default=False,
                        help="a flag to indicate whether use batch-norm for training")
    # added arguements for CoCoS
    parser.add_argument("--dis_layers", type=int, default=2,
                        help="the number of MLP discriminator layers, only for CoCoS-enhanced models")
    parser.add_argument("--emb_hid_dim", type=int, default=64,
                        help="the hidden dimension of the hidden layers in the MLP discriminator")
    parser.add_argument('--beta', type=float, default=0.6, help='weight for cocos contrastive loss')
    parser.add_argument("--cocos_cls_mode", type=str, default='both',
                        help="the type of the classification loss")

    # extra added arguements
    parser.add_argument("--seed", type=int, default=0,
                        help="the random seed to reproduce the result")

    parser.add_argument("--consistency_weight", type=float, default=0.1,)

    args = parser.parse_args()

    main(args)