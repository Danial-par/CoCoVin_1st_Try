import argparse
import torch
import sys
import os
import models
import models_ogb
import trainers
import torch_geometric as pyg
from torch_geometric.datasets import Planetoid
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

def load_data(db, db_dir='./dataset'):
    if db in ['Cora', 'CiteSeer', 'PubMed']:
        data = Planetoid(db_dir, db, transform=transforms.NormalizeFeatures())
        g = data[0]
        out_dim = data.num_classes
    elif db == 'ogbn-arxiv':
        data = PygNodePropPredDataset(name=db, root=db_dir, transform=transforms.NormalizeFeatures())
        g = data[0]
        out_dim = data.num_classes

        # Convert OGB format to standard format
        split_idx = data.get_idx_split()
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']

        # Add reverse edges to make the graph undirected
        edge_index = g.edge_index
        reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
        edge_index = torch.cat([edge_index, reverse_edge_index], dim=1)
        # Remove duplicate edges
        g.edge_index = pyg.utils.coalesce(edge_index)

        # Create boolean masks (standard format)
        num_nodes = g.num_nodes
        g.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        g.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        g.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Ensure labels are properly shaped
        g.y = g.y.squeeze()

        g.train_mask[train_idx] = True
        g.val_mask[val_idx] = True
        g.test_mask[test_idx] = True

        # Store original indices for OGB evaluation
        g.train_idx = train_idx
        g.val_idx = val_idx
        g.test_idx = test_idx
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
        if args.model in backbone_list:
            # For pure backbone models, use OGB version if needed
            model = getattr(models_ogb if args.dataset == 'ogbn-arxiv' else models, args.model)(info_dict)
        elif args.model.startswith('CoCoVin') or args.model.startswith('Violin'):
            # For composite models, always use regular models module
            model = getattr(models, args.model)(info_dict)
        else:
            # Default case
            model = getattr(models, args.model)(info_dict)

        # Initialize trainer
        if args.model in backbone_list:
            if args.dataset == 'ogbn-arxiv':
                trainer = getattr(trainers, 'BaseArxivTrainer')(g, model, info_dict)
            else:
                trainer = getattr(trainers, 'BaseTrainer')(g, model, info_dict)
        elif args.model.startswith('CoCoVin'):
            info_dict.update({'backbone': args.model[7:]})
            Dis = getattr(models_ogb if args.dataset == 'ogbn-arxiv' else models, 'DisMLP')(info_dict)
            Dis.to(info_dict['device'])
            if args.dataset == 'ogbn-arxiv':
                trainer = getattr(trainers, 'CoCoVinArxivTrainer')(g, model, info_dict, Dis=Dis)
            else:
                trainer = getattr(trainers, 'CoCoVinTrainer')(g, model, info_dict, Dis=Dis)
        elif args.model.startswith('Violin'):
            info_dict.update({'backbone': args.model[6:]})
            if args.dataset == 'ogbn-arxiv':
                trainer = getattr(trainers, 'ViolinArxivTrainer')(g, model, info_dict)
            else:
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
                        help="the dataset for the experiment")
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

    args = parser.parse_args()

    main(args)