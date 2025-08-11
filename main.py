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

def load_data(db, db_dir='./dataset'):
    if db in ['Cora', 'CiteSeer', 'PubMed']:
        data = Planetoid(db_dir, db, transform=transforms.NormalizeFeatures())
        g = data[0]
        out_dim = data.num_classes
    elif db == 'ogbn-arxiv':
        g, out_dim = ogb_prep('ogbn-arxiv', db_dir)
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

    # For CoCoVin models with tuning
    if args.model.startswith('CoCoVin') and args.tune:
        acc_list = []
        time_cost_list = []

        for i in range(args.round):
            print(f"\n=== ROUND {i + 1}/{args.round} ===\n")
            set_seed(args.seed + i)

            tic = time.time()
            val_acc, tt_acc = run_cocovine_with_tuning(args)
            toc = time.time()

            acc_list.append(tt_acc)
            time_cost = toc - tic
            time_cost_list.append(time_cost)
            print(f'Round {i + 1} complete. Time cost: {time_cost:.2f}s, Test acc: {tt_acc:.4f}')

        print('\n\n')
        print(f'The averaged accuracy of {args.round} rounds on {args.dataset} is: {np.mean(acc_list):.4f}')
        print(f'The averaged time cost of {args.round} rounds is {np.mean(time_cost_list):.4f}s')
        return

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


def run_cocovine_with_tuning(args):
    # Create a directory for results if it doesn't exist
    results_dir = os.path.join('exp', 'results')
    os.makedirs(results_dir, exist_ok=True)

    # First prepare the data
    g, info_dict = load_data(args.dataset)
    set_seed(args.seed)
    info_dict.update(args.__dict__)
    info_dict.update({'device': torch.device('cpu') if args.gpu == -1 else torch.device('cuda:{}'.format(args.gpu)),})

    # Step 1: Train the base model (backbone)
    print("\n--- STEP 1: Training base backbone model ---\n")
    backbone_name = args.model[7:]  # Strip "CoCoVin" prefix
    backbone_model = getattr(models_ogb if args.dataset == 'ogbn-arxiv' else models, backbone_name)(info_dict)
    backbone_model.to(info_dict['device'])
    backbone_trainer = getattr(trainers, 'BaseTrainer')(g, backbone_model, info_dict)
    val_acc, tt_acc, _, _, _, _ = backbone_trainer.train()
    print(f"Base model trained: val_acc={val_acc:.4f}, test_acc={tt_acc:.4f}")

    # Step 2: Train Phase 1 (CoCoS)
    print("\n--- STEP 2: Training Phase 1 (CoCoS) ---\n")
    model = getattr(models, args.model)(info_dict)
    model.to(info_dict['device'])
    info_dict.update({'backbone': args.model[7:]})
    Dis = getattr(models_ogb if args.dataset == 'ogbn-arxiv' else models, 'DisMLP')(info_dict)
    Dis.to(info_dict['device'])

    # Adjust epochs for phase 1
    original_epochs = args.n_epochs
    info_dict['n_epochs'] = args.n_epochs // 2

    # Create trainer and train phase 1
    trainer = getattr(trainers, 'CoCoVinTrainer')(g, model, info_dict, Dis=Dis)
    val_acc, tt_acc, _, _, _, _ = trainer.train()
    print(f"Phase 1 trained: val_acc={val_acc:.4f}, test_acc={tt_acc:.4f}")

    # Get the phase 1 model path
    phase1_model_path = os.path.join('exp', f"{args.model}_phase1", args.dataset,
                                    f"{args.model}_{args.dataset}_{args.seed}_phase1.pt")

    # Step 3: Hyperparameter tuning for Phase 2 (Violin)
    print("\n--- STEP 3: Hyperparameter Tuning for Phase 2 (Violin) ---\n")
    tuning_epochs = 300  # Number of epochs for each tuning run

    # Create fresh model and discriminator for tuning
    model_tune = getattr(models, args.model)(info_dict)
    model_tune.to(info_dict['device'])
    Dis_tune = getattr(models_ogb if args.dataset == 'ogbn-arxiv' else models, 'DisMLP')(info_dict)
    Dis_tune.to(info_dict['device'])

    # Run hyperparameter tuning
    best_params = trainers.hyperparameter_tune_violin(
        g, model_tune, Dis_tune, phase1_model_path, info_dict, tuning_epochs)

    # Step 4: Train with best hyperparameters for full run
    print("\n--- STEP 4: Final Training with Best Hyperparameters ---\n")

    # Create fresh model and discriminator for final training
    model_final = getattr(models, args.model)(info_dict)
    model_final.to(info_dict['device'])
    Dis_final = getattr(models_ogb if args.dataset == 'ogbn-arxiv' else models, 'DisMLP')(info_dict)
    Dis_final.to(info_dict['device'])

    # Update info_dict with best hyperparameters
    info_dict.update(best_params)
    info_dict['n_epochs'] = original_epochs  # Restore original number of epochs

    # Create trainer and load phase 1 state
    final_trainer = getattr(trainers, 'CoCoVinTrainer')(g, model_final, info_dict, Dis=Dis_final)

    # Load phase 1 checkpoint
    checkpoint = torch.load(phase1_model_path, map_location=info_dict['device'])
    model_final.load_state_dict(checkpoint['model'])
    Dis_final.load_state_dict(checkpoint['discriminator'])
    final_trainer.pred_labels = checkpoint['pred_labels']
    final_trainer.pred_conf = checkpoint['pred_conf']
    final_trainer.conf_thrs = checkpoint['conf_thrs']
    final_trainer.pred_label_flag = False

    # Skip phase 1
    final_trainer.phase1_epochs = 0

    # Train phase 2 only with best hyperparameters
    val_acc, tt_acc, val_acc_fin, tt_acc_fin, microf1, macrof1 = final_trainer.train(phase2_only=True)

    print(f"Final results with best hyperparameters: {best_params}")
    print(f"Best validation accuracy: {val_acc:.4f}")
    print(f"Best test accuracy: {tt_acc:.4f}")
    print(f"Micro-F1: {microf1:.4f}, Macro-F1: {macrof1:.4f}")

    return val_acc, tt_acc

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
    parser.add_argument("--tune", action='store_true', default=False,
                        help="enable hyperparameter tuning for CoCoVin models")

    args = parser.parse_args()

    main(args)