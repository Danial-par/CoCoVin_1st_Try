import torch
import numpy as np
from torch import nn
import os
import time
from copy import deepcopy
from sklearn import metrics
import torch_geometric as pyg
import matplotlib.pyplot as plt


def plot_results(tr_acc_history, val_acc_history, tt_acc_history, tr_loss_history, val_loss_history, tt_loss_history):
    """
    Plots the training, validation, and test accuracy and loss over epochs.
    """
    epochs = range(1, len(tr_acc_history) + 1)

    # Plotting Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, tr_acc_history, 'b', label='Training Acc')
    plt.plot(epochs, val_acc_history, 'g', label='Validation Acc')
    plt.plot(epochs, tt_acc_history, 'r', label='Test Acc')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plotting Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, tr_loss_history, 'b', label='Training Loss')
    plt.plot(epochs, val_loss_history, 'g', label='Validation Loss')
    plt.plot(epochs, tt_loss_history, 'r', label='Test Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

class BaseTrainer(object):
    '''
    Basic trainer for training.
    '''

    def __init__(self, g, model, info_dict, *args, **kwargs):
        self.g = g

        self.model = model
        self.info_dict = info_dict

        # load train/val/test split
        self.tr_mask = g.train_mask
        self.val_mask = g.val_mask
        self.tt_mask = g.test_mask
        self.tr_nid = g.train_mask.nonzero().squeeze()
        self.val_nid = g.val_mask.nonzero().squeeze()
        self.tt_nid = g.test_mask.nonzero().squeeze()
        self.labels = g.y
        self.tr_y = self.labels[self.tr_nid]
        self.val_y = self.labels[self.val_nid]
        self.tt_y = self.labels[self.tt_nid]
        self.ori_edge_index = g.edge_index.detach()

        self.crs_entropy_fn = nn.CrossEntropyLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=info_dict['lr'], weight_decay=info_dict['weight_decay'])

        self.best_val_acc = 0
        self.best_tt_acc = 0
        self.best_microf1 = 0
        self.best_macrof1 = 0

    def train(self):
        for i in range(self.info_dict['n_epochs']):
            tr_loss_epoch, tr_acc, tr_microf1, tr_macrof1 = self.train_epoch(i)
            (val_loss_epoch, val_acc_epoch, val_microf1_epoch, val_macrof1_epoch), \
            (tt_loss_epoch, tt_acc_epoch, tt_microf1_epoch, tt_macrof1_epoch) = self.eval_epoch(i)
            if val_acc_epoch > self.best_val_acc:
                self.best_val_acc = val_acc_epoch
                self.best_tt_acc = tt_acc_epoch
                self.best_microf1 = tt_microf1_epoch
                self.best_macrof1 = tt_macrof1_epoch
                _ = self.save_model(self.model, self.info_dict)

            if i % 100 == 0:
                print("Best val acc: {:.4f}, test acc: {:.4f}, micro-F1: {:.4f}, macro-F1: {:.4f}\n"
                      .format(self.best_val_acc, self.best_tt_acc, self.best_microf1, self.best_macrof1))

        # save the model in the final epoch
        _ = self.save_model(self.model, self.info_dict, state='fin')
        return self.best_val_acc, self.best_tt_acc, val_acc_epoch, tt_acc_epoch, self.best_microf1, self.best_macrof1

    def train_epoch(self, epoch_i):
        # training sample indices and labels
        nids = self.tr_nid
        labels = self.tr_y

        tic = time.time()
        self.model.train()
        labels = labels.to(self.info_dict['device'])
        with torch.set_grad_enabled(True):
            x_data = self.g.x.to(self.info_dict['device'])
            edge_index = self.g.edge_index.to(self.info_dict['device'])
            logits = self.model(x_data, edge_index)
            epoch_loss = self.crs_entropy_fn(logits[nids], labels)

            self.opt.zero_grad()
            epoch_loss.backward()
            self.opt.step()

            _, preds = torch.max(logits[nids], dim=1)
            epoch_acc = torch.sum(preds == labels).cpu().item() * 1.0 / labels.shape[0]
            epoch_micro_f1 = metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="micro")
            epoch_macro_f1 = metrics.f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

        toc = time.time()
        # if epoch_i % 10 == 0:
        #     print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.cpu().item(), epoch_acc))
        #     print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1, epoch_macro_f1))
        #     print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.cpu().item(), epoch_acc, epoch_micro_f1, epoch_macro_f1

    def eval_epoch(self, epoch_i):
        tic = time.time()
        self.model.eval()
        val_labels = self.val_y.to(self.info_dict['device'])
        tt_labels = self.tt_y.to(self.info_dict['device'])
        with torch.set_grad_enabled(False):
            x_data = self.g.x.to(self.info_dict['device'])
            edge_index = self.g.edge_index.to(self.info_dict['device'])
            logits = self.model(x_data, edge_index)
            val_epoch_loss = self.crs_entropy_fn(logits[self.val_nid], val_labels)
            tt_epoch_loss = self.crs_entropy_fn(logits[self.tt_nid], tt_labels)

            _, val_preds = torch.max(logits[self.val_nid], dim=1)
            _, tt_preds = torch.max(logits[self.tt_nid], dim=1)

            val_epoch_acc = torch.sum(val_preds == val_labels).cpu().item() * 1.0 / val_labels.shape[0]
            tt_epoch_acc = torch.sum(tt_preds == tt_labels).cpu().item() * 1.0 / tt_labels.shape[0]
            val_epoch_micro_f1 = metrics.f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average="micro")
            val_epoch_macro_f1 = metrics.f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average="macro")
            tt_epoch_micro_f1 = metrics.f1_score(tt_labels.cpu().numpy(), tt_preds.cpu().numpy(), average="micro")
            tt_epoch_macro_f1 = metrics.f1_score(tt_labels.cpu().numpy(), tt_preds.cpu().numpy(), average="macro")

        # toc = time.time()
        # if epoch_i % 10 == 0:
        #     print("Epoch {} | Loss: {:.4f} | validation accuracy: {:.4f}".format(epoch_i, val_epoch_loss.cpu().item(),
        #                                                                          val_epoch_acc))
        #     print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(val_epoch_micro_f1, val_epoch_macro_f1))
        #     print("Epoch {} | Loss: {:.4f} | testing accuracy: {:.4f}".format(epoch_i, tt_epoch_loss.cpu().item(),
        #                                                                          tt_epoch_acc))
        #     print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(tt_epoch_micro_f1, tt_epoch_macro_f1))
        #     print('Elapse time: {:.4f}s'.format(toc - tic))
        return (val_epoch_loss.cpu().item(), val_epoch_acc, val_epoch_micro_f1, val_epoch_macro_f1), \
               (tt_epoch_loss.cpu().item(), tt_epoch_acc, tt_epoch_micro_f1, tt_epoch_macro_f1)

    def save_model(self, model, info_dict, state='val'):

        save_root = os.path.join('exp', info_dict['model'] + '_ori', info_dict['dataset'])
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        # checkpoint name
        ckpname = '{model}_{db}_{seed}_{state}.pt'. \
            format(model=info_dict['model'], db=info_dict['dataset'],
                   seed=info_dict['seed'],
                   state=state)
        savedir = os.path.join(save_root, ckpname)
        torch.save(model.state_dict(), savedir)
        return savedir


class ViolinTrainer(BaseTrainer):
    def __init__(self, g, model, info_dict, *args, **kwargs):
        super().__init__(g, model, info_dict, *args, **kwargs)

        self.pred_labels = None
        self.pred_conf = None
        self.conf_thrs = 0

        self.virt_edge_index = None  # virtual links indices

        self.best_pretr_val_acc = None
        self.pretr_model_dir = os.path.join('exp', self.info_dict['backbone'] + '_ori', self.info_dict['dataset'], '{model}_{db}_{seed}_{state}.pt'.format(model=self.info_dict['backbone'], db=self.info_dict['dataset'], seed=self.info_dict['seed'], state='val',))
        self.load_pretr_model()
        self.pred_label_flag = True  # a flag to indicate whether it needs to predict labels for the next epoch

        # Attributes for tracking training history
        self.tr_loss_history = []
        self.tr_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.tt_loss_history = []
        self.tt_acc_history = []

    def load_pretr_model(self):
        # load learnable weights from the pretrained model
        self.model.load_state_dict(torch.load(self.pretr_model_dir, map_location=self.info_dict['device']))

    def add_VOs(self):
        '''
        add virtual links based on the ground-truth class labels, within the whole graph
        :return:
        '''
        labels = self.pred_labels.cpu()
        conf = self.pred_conf.cpu()
        conf_mask = conf >= self.conf_thrs

        ori_adj = self.ori_edge_index
        # print('The number of edges before adding virtual links: {}'.format(ori_adj.shape[1]))

        # number of virtual links to be added for each node
        n_vo = self.info_dict['m']
        virt_edges = []
        tr_mask = self.g.train_mask.bool()
        other_mask = ~tr_mask
        # construct virtual links
        label_list = list(set(labels.numpy().tolist()))
        for i in range(n_vo):
            dsts = torch.arange(self.g.num_nodes)
            srcs = -1 * torch.ones_like(dsts)
            for k in label_list:
                # the indices of nodes that are from the k-th class
                k_mask = labels == k
                tr_k_mask = k_mask * tr_mask  # training nodes in the k-th class
                other_k_mask = k_mask * other_mask  # unlabeled nodes that predicted to be in the k-th class

                # add virtual links for labeled nodes
                # randomly select nodes from the whole graph
                tr_vl_k_idx = torch.arange(self.g.num_nodes)[k_mask]
                tr_vl_rand_idx = torch.from_numpy(np.random.choice(tr_vl_k_idx, tr_k_mask.sum().item(), replace=True))
                srcs[tr_k_mask] = tr_vl_rand_idx

                # add virtual links for the unlabeled nodes
                other_vl_k_idx = torch.arange(self.g.num_nodes)[k_mask]
                other_vl_rand_idx = torch.from_numpy(np.random.choice(other_vl_k_idx, other_k_mask.sum().item(),
                                                                    replace=True))
                srcs[other_k_mask] = other_vl_rand_idx

            # qualified edges
            qua_mask = srcs >= 0
            qua_mask = conf_mask * qua_mask
            srcs = srcs[qua_mask]
            dsts = dsts[qua_mask]

            vl_i = torch.cat([srcs.unsqueeze(0), dsts.unsqueeze(0)], dim=0)
            virt_edges.append(vl_i)

        virt_edges = torch.cat(virt_edges, dim=1)
        rev_virt_edges = torch.cat([virt_edges[1].unsqueeze(0), virt_edges[0].unsqueeze(0)], dim=0)
        full_virt_edges = torch.cat((virt_edges, rev_virt_edges), dim=1)

        cur_adj = torch.cat((ori_adj, full_virt_edges), dim=1)
        cur_adj = pyg.utils.coalesce(cur_adj)
        # print('The number of edges after adding virtual links: {}'.format(cur_adj.shape[1]))
        self.g.edge_index = cur_adj
        self.virt_edge_index = full_virt_edges

    def train(self):

        for i in range(self.info_dict['n_epochs']):
            if i % self.info_dict['eta'] == 0:
                if self.pred_label_flag:
                    # progressively update/ override the predicted labels
                    self.get_pred_labels()
                # add virtual links based on the predicted labels
                self.add_VOs()

            tr_loss_epoch, tr_acc, tr_microf1, tr_macrof1 = self.train_epoch(i)
            (val_loss_epoch, val_acc_epoch, val_microf1_epoch, val_macrof1_epoch), \
            (tt_loss_epoch, tt_acc_epoch, tt_microf1_epoch, tt_macrof1_epoch) = self.eval_epoch(i)

            # Store metrics for plotting
            self.tr_loss_history.append(tr_loss_epoch)
            self.tr_acc_history.append(tr_acc)
            self.val_loss_history.append(val_loss_epoch)
            self.val_acc_history.append(val_acc_epoch)
            self.tt_loss_history.append(tt_loss_epoch)
            self.tt_acc_history.append(tt_acc_epoch)
            # =====================

            if val_acc_epoch > self.best_val_acc:
                self.best_val_acc = val_acc_epoch
                self.best_tt_acc = tt_acc_epoch
                self.best_microf1 = tt_microf1_epoch
                self.best_macrof1 = tt_macrof1_epoch
                save_model_dir = self.save_model(self.model, self.info_dict)
                if val_acc_epoch > self.best_pretr_val_acc:
                    # update the pretraining model's parameter directory, we will use the updated pretraining model to
                    # generate estimated labels in the following epochs
                    self.pretr_model_dir = save_model_dir
                    self.pred_label_flag = True
                    print(f"epoch {i:03d} | new best validation accuracy {self.best_val_acc:.4f} - test accuracy {self.best_tt_acc:.4f}")

            if i % 50 == 0:
                print(
                    f"Epoch {i:03d} | Loss: {tr_loss_epoch:.4f} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc_epoch:.4f} | Test Acc: {tt_acc_epoch:.4f}")
            # if i % self.info_dict['eta'] == 0:
            #     print("Best val acc: {:.4f}, test acc: {:.4f}, micro-F1: {:.4f}, macro-F1: {:.4f}\n"
            #           .format(self.best_val_acc, self.best_tt_acc, self.best_microf1, self.best_macrof1))

        # Plot results after training is complete
        plot_results(self.tr_acc_history, self.val_acc_history, self.tt_acc_history, self.tr_loss_history, self.val_loss_history, self.tt_loss_history)
        # =====================

        return self.best_val_acc, self.best_tt_acc, val_acc_epoch, tt_acc_epoch, self.best_microf1, self.best_macrof1

    def train_epoch(self, epoch_i):
        # training sample indices and labels, for the supervised loss
        cls_nids = self.tr_nid
        cls_labels = self.tr_y
        cls_labels = cls_labels.to(self.info_dict['device'])
        # unlabeled node indices for consistent learning, for the consistent loss
        con_nids = torch.cat((self.val_nid, self.tt_nid))

        tic = time.time()
        self.model.train()
        with torch.set_grad_enabled(True):
            x_data = self.g.x.to(self.info_dict['device'])
            ori_edge_index = self.ori_edge_index.to(self.info_dict['device'])
            aug_edge_index = self.g.edge_index.to(self.info_dict['device'])
            virt_edge_index = self.virt_edge_index.to(self.info_dict['device'])
            # results on the original graph
            ori_logits = self.model(x_data, ori_edge_index)
            ori_conf = torch.softmax(ori_logits, dim=1)
            # results on the semantic-consistent graph
            aug_logits = self.model(x_data, aug_edge_index)
            aug_conf = torch.softmax(aug_logits, dim=1)

            # consistent loss
            epoch_con_loss = torch.abs(ori_conf[con_nids] - aug_conf[con_nids]).sum(dim=1).mean()
            # virtual link loss
            num_unq_vls = virt_edge_index.shape[1] // 2
            epoch_vl_loss = torch.abs(aug_conf[virt_edge_index[0, :num_unq_vls]] -
                                      aug_conf[virt_edge_index[1, :num_unq_vls]]).sum(dim=1).mean()
            # classification loss
            if self.info_dict['cls_mode'] == 'ori':
                epoch_cls_loss = self.crs_entropy_fn(ori_logits[cls_nids], cls_labels)
                _, preds = torch.max(ori_logits[cls_nids], dim=1)
            elif self.info_dict['cls_mode'] == 'virt':
                epoch_cls_loss = self.crs_entropy_fn(aug_logits[cls_nids], cls_labels)
                _, preds = torch.max(aug_logits[cls_nids], dim=1)
            elif self.info_dict['cls_mode'] == 'both':
                epoch_cls_loss = 0.5 * (self.crs_entropy_fn(ori_logits[cls_nids], cls_labels) +
                                        self.crs_entropy_fn(aug_logits[cls_nids], cls_labels))
                _, preds = torch.max((ori_logits + aug_logits)[cls_nids], dim=1)
            else:
                raise ValueError("Unexpected cls_mode parameter: {}".format(self.info_dict['cls_mode']))

            epoch_loss = epoch_cls_loss + self.info_dict['alpha'] * epoch_con_loss + self.info_dict['gamma'] * epoch_vl_loss

            self.opt.zero_grad()
            epoch_loss.backward()
            self.opt.step()

            epoch_acc = torch.sum(preds == cls_labels).cpu().item() * 1.0 / cls_labels.shape[0]
            epoch_micro_f1 = metrics.f1_score(cls_labels.cpu().numpy(), preds.cpu().numpy(), average="micro")
            epoch_macro_f1 = metrics.f1_score(cls_labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

        # toc = time.time()
        # if epoch_i % 10 == 0:
        #     print("Epoch {} | Loss: {:.4f} | training accuracy: {:.4f}".format(epoch_i, epoch_loss.cpu().item(), epoch_acc))
        #     print("cls loss: {:.4f} | consistent loss: {:.4f} | vl loss: {:.4f} ".format(epoch_cls_loss.cpu().item(), epoch_con_loss.cpu().item(), epoch_vl_loss.cpu().item()))
        #     print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(epoch_micro_f1, epoch_macro_f1))
        #     print('Elapse time: {:.4f}s'.format(toc - tic))
        return epoch_loss.cpu().item(), epoch_acc, epoch_micro_f1, epoch_macro_f1

    def eval_epoch(self, epoch_i):
        tic = time.time()
        self.model.eval()
        val_labels = self.val_y.to(self.info_dict['device'])
        tt_labels = self.tt_y.to(self.info_dict['device'])
        with torch.set_grad_enabled(False):
            x_data = self.g.x.to(self.info_dict['device'])
            ori_edge_index = self.ori_edge_index.to(self.info_dict['device'])
            ori_logits = self.model(x_data, ori_edge_index)
            ori_conf = torch.softmax(ori_logits, dim=1)
            val_epoch_loss = self.crs_entropy_fn(ori_logits[self.val_nid], val_labels)
            tt_epoch_loss = self.crs_entropy_fn(ori_logits[self.tt_nid], tt_labels)
            _, val_preds = torch.max(ori_logits[self.val_nid], dim=1)
            _, tt_preds = torch.max(ori_logits[self.tt_nid], dim=1)

            val_epoch_acc = torch.sum(val_preds == val_labels).cpu().item() * 1.0 / val_labels.shape[0]
            tt_epoch_acc = torch.sum(tt_preds == tt_labels).cpu().item() * 1.0 / tt_labels.shape[0]
            val_epoch_micro_f1 = metrics.f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average="micro")
            val_epoch_macro_f1 = metrics.f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average="macro")
            tt_epoch_micro_f1 = metrics.f1_score(tt_labels.cpu().numpy(), tt_preds.cpu().numpy(), average="micro")
            tt_epoch_macro_f1 = metrics.f1_score(tt_labels.cpu().numpy(), tt_preds.cpu().numpy(), average="macro")

        # toc = time.time()
        # if epoch_i % 10 == 0:
        #     print("Epoch {} | validation loss: {:.4f} | validation accuracy: {:.4f}".format(epoch_i, val_epoch_loss.cpu().item(), val_epoch_acc))
        #     print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(val_epoch_micro_f1, val_epoch_macro_f1))
        #     print("Epoch {} | test loss: {:.4f} | testing accuracy: {:.4f}".format(epoch_i, tt_epoch_loss.cpu().item(), tt_epoch_acc))
        #     print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(tt_epoch_micro_f1, tt_epoch_macro_f1))
        #     print('Elapse time: {:.4f}s'.format(toc - tic))
        return (val_epoch_loss.cpu().item(), val_epoch_acc, val_epoch_micro_f1, val_epoch_macro_f1), \
               (tt_epoch_loss.cpu().item(), tt_epoch_acc, tt_epoch_micro_f1, tt_epoch_macro_f1)

    def get_pred_labels(self):

        # load the pretrained model and use it to estimate the labels
        cur_model_state_dict = deepcopy(self.model.state_dict())
        self.model.load_state_dict(torch.load(self.pretr_model_dir, map_location=self.info_dict['device']))

        self.model.eval()
        with torch.set_grad_enabled(False):
            x_data = self.g.x.to(self.info_dict['device'])
            edge_index = self.ori_edge_index.to(self.info_dict['device'])
            logits = self.model(x_data, edge_index)

            _, preds = torch.max(logits, dim=1)
            conf = torch.softmax(logits, dim=1).max(dim=1)[0]
            self.pred_labels = preds
            # for training nodes, the estimated labels will be replaced by their ground-truth labels
            self.pred_labels[self.tr_mask] = self.labels[self.tr_mask].to(self.info_dict['device'])
            self.pred_conf = conf

            pretr_val_acc = torch.sum(preds[self.val_nid].cpu() == self.labels[self.val_nid]).item() * 1.0 / \
                            self.labels[self.val_nid].shape[0]
            self.best_pretr_val_acc = pretr_val_acc

            # set the confidence threshold
            self.set_conf_thrs(preds, conf, self.val_nid)

        # reload the current model's parameters
        self.model.load_state_dict(cur_model_state_dict)
        self.pred_label_flag = False

    def set_conf_thrs(self, preds, conf, nids):

        # calculate the confidence threshold based on the validation set for adding virtual links
        preds, conf = preds.cpu(), conf.cpu()
        val_conf = (conf[nids] * 10).long().cpu().numpy()
        val_correct_mask = preds[nids] == self.labels[nids]
        val_wrong_mask = ~val_correct_mask
        val_correct_conf = val_conf[val_correct_mask]
        val_wrong_conf = val_conf[val_wrong_mask]

        # the distribution of validation predictions of each confidence interval
        val_correct_dist = np.bincount(val_correct_conf)
        if len(val_correct_dist) < 11:  # pad the dist vect if no high confidence predictions occur
            val_correct_dist = np.hstack((val_correct_dist, np.array([0] * (11 - len(val_correct_dist)))))
        val_wrong_dist = np.bincount(val_wrong_conf)
        if len(val_wrong_dist) < 11:
            val_wrong_dist = np.hstack((val_wrong_dist, np.array([0] * (11 - len(val_wrong_dist)))))
        val_dist = np.bincount(val_conf)
        if len(val_dist) < 11:
            val_dist = np.hstack((val_dist, np.array([0] * (11 - len(val_dist)))))

        # calculate the validation accuracies of each confidence interval
        val_correct_cumsum = val_correct_dist[::-1].cumsum()[::-1]
        val_wrong_cumsum = val_wrong_dist[::-1].cumsum()[::-1]
        val_cumsum = val_dist[::-1].cumsum()[::-1]
        val_conf_acc = val_correct_cumsum / val_cumsum
        val_conf_acc[np.isnan(val_conf_acc)] = 0  # zero the nan elements

        acc_thrs = self.info_dict['delta']
        # calculate the confidence threshold: find the first qualified confidence interval index
        try:
            # set the confidence threshold based on the given accuracy requirements
            self.conf_thrs = (np.where(val_conf_acc > acc_thrs)[0][0] / 10.).item()
            # print('The (dynamic) confidence threshold is: {:.4f}'.format(self.conf_thrs))
        except IndexError:  ## no confidence interval fulfills the accuracy requirement
            # find the confidence interval that with the highest acc
            self.conf_thrs = (np.argmax(val_conf_acc) / 10.).item()
            print('The confidence requirement is not fulfilled. Use the confidence level with the best acc.')
            print('The (workaround) confidence threshold is: {:.4f}'.format(self.conf_thrs))
        return



class CoCoVinTrainer(BaseTrainer):
    def __init__(self, g, model, info_dict, *args, **kwargs):
        super().__init__(g, model, info_dict, *args, **kwargs)

        # Attributes from ViolinTrainer
        self.pred_labels = None
        self.pred_conf = None
        self.conf_thrs = 0
        self.virt_edge_index = None
        self.best_pretr_val_acc = None
        self.pretr_model_dir = os.path.join('exp', self.info_dict['backbone'] + '_ori', self.info_dict['dataset'], '{model}_{db}_{seed}_{state}.pt'.format(model=self.info_dict['backbone'], db=self.info_dict['dataset'], seed=self.info_dict['seed'], state='val',))
        self.load_pretr_model()
        self.pred_label_flag = True

        # Attributes from CoCoSTrainer
        self.Dis = kwargs['Dis']
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam([{'params': self.model.parameters()},
                                     {'params': self.Dis.parameters()}],
                                    lr=info_dict['lr'], weight_decay=info_dict['weight_decay'])

        # Attributes for tracking training history
        self.tr_loss_history = []
        self.tr_acc_history = []
        self.val_loss_history = []
        self.val_acc_history = []
        self.tt_loss_history = []
        self.tt_acc_history = []

    def load_pretr_model(self):
        self.model.load_state_dict(torch.load(self.pretr_model_dir, map_location=self.info_dict['device']))

    def train(self):
        for i in range(self.info_dict['n_epochs']):
            if i % self.info_dict['eta'] == 0:
                if self.pred_label_flag:
                    self.get_pred_labels()
                self.add_VOs()

            tr_loss_epoch, tr_acc, tr_microf1, tr_macrof1 = self.train_epoch(i)
            (val_loss_epoch, val_acc_epoch, val_microf1_epoch, val_macrof1_epoch), \
            (tt_loss_epoch, tt_acc_epoch, tt_microf1_epoch, tt_macrof1_epoch) = self.eval_epoch(i)

            # === ADD THIS PART ===
            # Store metrics for plotting
            self.tr_loss_history.append(tr_loss_epoch)
            self.tr_acc_history.append(tr_acc)
            self.val_loss_history.append(val_loss_epoch)
            self.val_acc_history.append(val_acc_epoch)
            self.tt_loss_history.append(tt_loss_epoch)
            self.tt_acc_history.append(tt_acc_epoch)
            # =====================

            if val_acc_epoch > self.best_val_acc:
                self.best_val_acc = val_acc_epoch
                self.best_tt_acc = tt_acc_epoch
                self.best_microf1 = tt_microf1_epoch
                self.best_macrof1 = tt_macrof1_epoch
                save_model_dir = self.save_model(self.model, self.info_dict)
                if val_acc_epoch > self.best_pretr_val_acc:
                    self.pretr_model_dir = save_model_dir
                    self.pred_label_flag = True

                print(f"epoch {i:03d} | new best validation accuracy {self.best_val_acc:.4f} - test accuracy {self.best_tt_acc:.4f}")

            if i % 50 == 0:
                print(
                    f"Epoch {i:03d} | Loss: {tr_loss_epoch:.4f} | Train Acc: {tr_acc:.4f} | Val Acc: {val_acc_epoch:.4f} | Test Acc: {tt_acc_epoch:.4f}")

            # if i % self.info_dict['eta'] == 0:
            #     print("Training acc: {:.4f}, val acc: {:.4f}, test acc: {:.4f}, micro-F1: {:.4f}, macro-F1: {:.4f}\n"
            #           .format(tr_acc, val_acc_epoch, tt_acc_epoch, tr_microf1, tr_macrof1))
            #     print("___________________________________________________________________________________")
            #     print("Best val acc: {:.4f}, test acc: {:.4f}, micro-F1: {:.4f}, macro-F1: {:.4f}\n"
            #           .format(self.best_val_acc, self.best_tt_acc, self.best_microf1, self.best_macrof1))

        # Plot results after training is complete
        plot_results(self.tr_acc_history, self.val_acc_history, self.tt_acc_history, self.tr_loss_history,
                          self.val_loss_history, self.tt_loss_history)
        # =====================

        return self.best_val_acc, self.best_tt_acc, val_acc_epoch, tt_acc_epoch, self.best_microf1, self.best_macrof1

    def train_epoch(self, epoch_i):
            # Violin training setup
            cls_nids = self.tr_nid
            cls_labels = self.tr_y.to(self.info_dict['device'])
            con_nids = torch.cat((self.val_nid, self.tt_nid))

            # CoCoS training setup
            ctr_nids = torch.cat((self.val_nid, self.tt_nid))
            ctr_labels_pos = torch.ones_like(ctr_nids, device=self.info_dict['device']).unsqueeze(dim=-1).float()
            ctr_labels_neg = torch.zeros_like(ctr_nids, device=self.info_dict['device']).unsqueeze(dim=-1).float()

            tic = time.time()
            self.model.train()
            self.Dis.train()
            with torch.set_grad_enabled(True):
                # Violin forward passes
                x_data = self.g.x.to(self.info_dict['device'])
                ori_edge_index = self.ori_edge_index.to(self.info_dict['device'])
                aug_edge_index = self.g.edge_index.to(self.info_dict['device'])
                virt_edge_index = self.virt_edge_index.to(self.info_dict['device'])
                ori_logits = self.model(x_data, ori_edge_index)
                ori_conf = torch.softmax(ori_logits, dim=1)
                aug_logits = self.model(x_data, aug_edge_index)
                aug_conf = torch.softmax(aug_logits, dim=1)

                # CoCoS forward passes
                shuf_feat = self.shuffle_feat(x_data)
                shuf_logits = self.model(shuf_feat, ori_edge_index) # Using original graph structure
                tp_shuf_nids = self.shuffle_nids()
                tp_shuf_logits = shuf_logits[tp_shuf_nids]

                # Violin Loss Calculation
                epoch_con_loss = torch.abs(ori_conf[con_nids] - aug_conf[con_nids]).sum(dim=1).mean()
                num_unq_vls = virt_edge_index.shape[1] // 2
                epoch_vl_loss = torch.abs(aug_conf[virt_edge_index[0, :num_unq_vls]] - aug_conf[virt_edge_index[1, :num_unq_vls]]).sum(dim=1).mean()
                if self.info_dict['cls_mode'] == 'ori':
                    epoch_cls_loss = self.crs_entropy_fn(ori_logits[cls_nids], cls_labels)
                    _, preds = torch.max(ori_logits[cls_nids], dim=1)
                elif self.info_dict['cls_mode'] == 'virt':
                    epoch_cls_loss = self.crs_entropy_fn(aug_logits[cls_nids], cls_labels)
                    _, preds = torch.max(aug_logits[cls_nids], dim=1)
                elif self.info_dict['cls_mode'] == 'both':
                    epoch_cls_loss = 0.5 * (self.crs_entropy_fn(ori_logits[cls_nids], cls_labels) + self.crs_entropy_fn(aug_logits[cls_nids], cls_labels))
                    _, preds = torch.max((ori_logits + aug_logits)[cls_nids], dim=1)
                else:
                    raise ValueError("Unexpected cls_mode parameter: {}".format(self.info_dict['cls_mode']))

                # CoCoS Contrastive Loss Calculation ('FS' mode)
                # 'F' mode positive pairs
                pos_score_f = self.Dis(torch.cat((shuf_logits, ori_logits), dim=-1))
                pos_loss_f = self.bce_fn(pos_score_f[ctr_nids], ctr_labels_pos)
                # 'S' mode positive pairs
                pos_score_s = self.Dis(torch.cat((tp_shuf_logits, shuf_logits), dim=-1))
                pos_loss_s = self.bce_fn(pos_score_s[ctr_nids], ctr_labels_pos)

                epoch_ctr_loss_pos = (pos_loss_f + pos_loss_s) / 2.0

                # Negative pairs
                neg_nids = self.gen_neg_nids()
                neg_ori_logits = ori_logits[neg_nids].detach()
                neg_score = self.Dis(torch.cat((ori_logits, neg_ori_logits), dim=-1))
                epoch_ctr_loss_neg = self.bce_fn(neg_score[ctr_nids], ctr_labels_neg)

                epoch_ctr_loss = epoch_ctr_loss_pos + epoch_ctr_loss_neg

                # Combined Loss
                epoch_loss = epoch_cls_loss + self.info_dict['alpha'] * epoch_con_loss + self.info_dict['gamma'] * epoch_vl_loss + self.info_dict['beta'] * epoch_ctr_loss

                self.opt.zero_grad()
                epoch_loss.backward()
                self.opt.step()

                epoch_acc = torch.sum(preds == cls_labels).cpu().item() * 1.0 / cls_labels.shape[0]
                epoch_micro_f1 = metrics.f1_score(cls_labels.cpu().numpy(), preds.cpu().numpy(), average="micro")
                epoch_macro_f1 = metrics.f1_score(cls_labels.cpu().numpy(), preds.cpu().numpy(), average="macro")

            # toc = time.time()
            # if epoch_i % 10 == 0:
            #     print(f"Epoch {epoch_i} | Loss: {epoch_loss.item():.4f} | Train Acc: {epoch_acc:.4f}")
            #     print(f"Cls: {epoch_cls_loss.item():.4f} | Con: {epoch_con_loss.item():.4f} | VL: {epoch_vl_loss.item():.4f} | Ctr: {epoch_ctr_loss.item():.4f}")
            #     print(f"Micro-F1: {epoch_micro_f1:.4f} | Macro-F1: {epoch_macro_f1:.4f}")
            #     print(f"Elapse time: {toc - tic:.4f}s")
            return epoch_loss.cpu().item(), epoch_acc, epoch_micro_f1, epoch_macro_f1

    # --- Include all helper methods from ViolinTrainer ---
    # add_VOs, eval_epoch, get_pred_labels, set_conf_thrs
    def add_VOs(self):
        '''
        add virtual links based on the ground-truth class labels, within the whole graph
        :return:
        '''
        labels = self.pred_labels.cpu()
        conf = self.pred_conf.cpu()
        conf_mask = conf >= self.conf_thrs

        ori_adj = self.ori_edge_index
        # print('The number of edges before adding virtual links: {}'.format(ori_adj.shape[1]))

        # number of virtual links to be added for each node
        n_vo = self.info_dict['m']
        virt_edges = []
        tr_mask = self.g.train_mask.bool()
        other_mask = ~tr_mask
        # construct virtual links
        label_list = list(set(labels.numpy().tolist()))
        for i in range(n_vo):
            dsts = torch.arange(self.g.num_nodes)
            srcs = -1 * torch.ones_like(dsts)
            for k in label_list:
                # the indices of nodes that are from the k-th class
                k_mask = labels == k
                tr_k_mask = k_mask * tr_mask  # training nodes in the k-th class
                other_k_mask = k_mask * other_mask  # unlabeled nodes that predicted to be in the k-th class

                # add virtual links for labeled nodes
                # randomly select nodes from the whole graph
                tr_vl_k_idx = torch.arange(self.g.num_nodes)[k_mask]
                tr_vl_rand_idx = torch.from_numpy(np.random.choice(tr_vl_k_idx, tr_k_mask.sum().item(), replace=True))
                srcs[tr_k_mask] = tr_vl_rand_idx

                # add virtual links for the unlabeled nodes
                other_vl_k_idx = torch.arange(self.g.num_nodes)[k_mask]
                other_vl_rand_idx = torch.from_numpy(np.random.choice(other_vl_k_idx, other_k_mask.sum().item(),
                                                                    replace=True))
                srcs[other_k_mask] = other_vl_rand_idx

            # qualified edges
            qua_mask = srcs >= 0
            qua_mask = conf_mask * qua_mask
            srcs = srcs[qua_mask]
            dsts = dsts[qua_mask]

            vl_i = torch.cat([srcs.unsqueeze(0), dsts.unsqueeze(0)], dim=0)
            virt_edges.append(vl_i)

        virt_edges = torch.cat(virt_edges, dim=1)
        rev_virt_edges = torch.cat([virt_edges[1].unsqueeze(0), virt_edges[0].unsqueeze(0)], dim=0)
        full_virt_edges = torch.cat((virt_edges, rev_virt_edges), dim=1)

        cur_adj = torch.cat((ori_adj, full_virt_edges), dim=1)
        cur_adj = pyg.utils.coalesce(cur_adj)
        # print('The number of edges after adding virtual links: {}'.format(cur_adj.shape[1]))
        self.g.edge_index = cur_adj
        self.virt_edge_index = full_virt_edges

    def eval_epoch(self, epoch_i):
        tic = time.time()
        self.model.eval()
        val_labels = self.val_y.to(self.info_dict['device'])
        tt_labels = self.tt_y.to(self.info_dict['device'])
        with torch.set_grad_enabled(False):
            x_data = self.g.x.to(self.info_dict['device'])
            ori_edge_index = self.ori_edge_index.to(self.info_dict['device'])
            ori_logits = self.model(x_data, ori_edge_index)
            ori_conf = torch.softmax(ori_logits, dim=1)
            val_epoch_loss = self.crs_entropy_fn(ori_logits[self.val_nid], val_labels)
            tt_epoch_loss = self.crs_entropy_fn(ori_logits[self.tt_nid], tt_labels)
            _, val_preds = torch.max(ori_logits[self.val_nid], dim=1)
            _, tt_preds = torch.max(ori_logits[self.tt_nid], dim=1)

            val_epoch_acc = torch.sum(val_preds == val_labels).cpu().item() * 1.0 / val_labels.shape[0]
            tt_epoch_acc = torch.sum(tt_preds == tt_labels).cpu().item() * 1.0 / tt_labels.shape[0]
            val_epoch_micro_f1 = metrics.f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average="micro")
            val_epoch_macro_f1 = metrics.f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average="macro")
            tt_epoch_micro_f1 = metrics.f1_score(tt_labels.cpu().numpy(), tt_preds.cpu().numpy(), average="micro")
            tt_epoch_macro_f1 = metrics.f1_score(tt_labels.cpu().numpy(), tt_preds.cpu().numpy(), average="macro")

        toc = time.time()
        # if epoch_i % 10 == 0:
        #     print("Epoch {} | validation loss: {:.4f} | validation accuracy: {:.4f}".format(epoch_i, val_epoch_loss.cpu().item(), val_epoch_acc))
        #     print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(val_epoch_micro_f1, val_epoch_macro_f1))
        #     print("Epoch {} | test loss: {:.4f} | testing accuracy: {:.4f}".format(epoch_i, tt_epoch_loss.cpu().item(), tt_epoch_acc))
        #     print("Micro-F1: {:.4f} | Macro-F1: {:.4f}".format(tt_epoch_micro_f1, tt_epoch_macro_f1))
        #     print('Elapse time: {:.4f}s'.format(toc - tic))
        return (val_epoch_loss.cpu().item(), val_epoch_acc, val_epoch_micro_f1, val_epoch_macro_f1), \
               (tt_epoch_loss.cpu().item(), tt_epoch_acc, tt_epoch_micro_f1, tt_epoch_macro_f1)

    def get_pred_labels(self):

        # load the pretrained model and use it to estimate the labels
        cur_model_state_dict = deepcopy(self.model.state_dict())
        self.model.load_state_dict(torch.load(self.pretr_model_dir, map_location=self.info_dict['device']))

        self.model.eval()
        with torch.set_grad_enabled(False):
            x_data = self.g.x.to(self.info_dict['device'])
            edge_index = self.ori_edge_index.to(self.info_dict['device'])
            logits = self.model(x_data, edge_index)

            _, preds = torch.max(logits, dim=1)
            conf = torch.softmax(logits, dim=1).max(dim=1)[0]
            self.pred_labels = preds
            # for training nodes, the estimated labels will be replaced by their ground-truth labels
            self.pred_labels[self.tr_mask] = self.labels[self.tr_mask].to(self.info_dict['device'])
            self.pred_conf = conf

            pretr_val_acc = torch.sum(preds[self.val_nid].cpu() == self.labels[self.val_nid]).item() * 1.0 / \
                            self.labels[self.val_nid].shape[0]
            self.best_pretr_val_acc = pretr_val_acc

            # set the confidence threshold
            self.set_conf_thrs(preds, conf, self.val_nid)

        # reload the current model's parameters
        self.model.load_state_dict(cur_model_state_dict)
        self.pred_label_flag = False

    def set_conf_thrs(self, preds, conf, nids):

        # calculate the confidence threshold based on the validation set for adding virtual links
        preds, conf = preds.cpu(), conf.cpu()
        val_conf = (conf[nids] * 10).long().cpu().numpy()
        val_correct_mask = preds[nids] == self.labels[nids]
        val_wrong_mask = ~val_correct_mask
        val_correct_conf = val_conf[val_correct_mask]
        val_wrong_conf = val_conf[val_wrong_mask]

        # the distribution of validation predictions of each confidence interval
        val_correct_dist = np.bincount(val_correct_conf)
        if len(val_correct_dist) < 11:  # pad the dist vect if no high confidence predictions occur
            val_correct_dist = np.hstack((val_correct_dist, np.array([0] * (11 - len(val_correct_dist)))))
        val_wrong_dist = np.bincount(val_wrong_conf)
        if len(val_wrong_dist) < 11:
            val_wrong_dist = np.hstack((val_wrong_dist, np.array([0] * (11 - len(val_wrong_dist)))))
        val_dist = np.bincount(val_conf)
        if len(val_dist) < 11:
            val_dist = np.hstack((val_dist, np.array([0] * (11 - len(val_dist)))))

        # calculate the validation accuracies of each confidence interval
        val_correct_cumsum = val_correct_dist[::-1].cumsum()[::-1]
        val_wrong_cumsum = val_wrong_dist[::-1].cumsum()[::-1]
        val_cumsum = val_dist[::-1].cumsum()[::-1]
        val_conf_acc = val_correct_cumsum / val_cumsum
        val_conf_acc[np.isnan(val_conf_acc)] = 0  # zero the nan elements

        acc_thrs = self.info_dict['delta']
        # calculate the confidence threshold: find the first qualified confidence interval index
        try:
            # set the confidence threshold based on the given accuracy requirements
            self.conf_thrs = (np.where(val_conf_acc > acc_thrs)[0][0] / 10.).item()
            # print('The (dynamic) confidence threshold is: {:.4f}'.format(self.conf_thrs))
        except IndexError:  ## no confidence interval fulfills the accuracy requirement
            # find the confidence interval that with the highest acc
            self.conf_thrs = (np.argmax(val_conf_acc) / 10.).item()
            print('The confidence requirement is not fulfilled. Use the confidence level with the best acc.')
            print('The (workaround) confidence threshold is: {:.4f}'.format(self.conf_thrs))
        return

    # --- Include all helper methods from CoCoSTrainer ---
    # shuffle_feat, shuffle_nids, gen_neg_nids

    def shuffle_feat(self, nfeat):
        pos_feat = nfeat.clone().detach()

        nid = torch.arange(self.g.num_nodes, device=self.info_dict['device'])
        labels = self.pred_labels
        if labels == None:
            raise ValueError('The class of unlabeled nodes have not been estimated!')

        # generate positive features
        shuf_nid = torch.zeros_like(nid).to(self.info_dict['device'])
        for i in range(self.info_dict['out_dim']):
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]
            # node ids with label class i
            i_nid = nid[i_pos]
            # shuffle the i-th class node ids
            i_shuffle_ind = torch.randperm(len(i_pos)).to(self.info_dict['device'])
            i_nid_shuffled = i_nid[i_shuffle_ind]
            # get new id arrangement for the i-th class
            shuf_nid[i_pos] = i_nid_shuffled.to(self.info_dict['device'])
        pos_feat[nid] = nfeat[shuf_nid].detach()

        return pos_feat

    def shuffle_nids(self):
        nid = torch.arange(self.g.num_nodes, device=self.info_dict['device'])
        labels = self.pred_labels
        if labels == None:
            raise ValueError('The class of unlabeled nodes have not been estimated!')

        # randomly sample a positive counterpart for each node
        shuf_nid = torch.arange(self.g.num_nodes).to(self.info_dict['device'])
        for i in range(self.info_dict['out_dim']):
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0].to(self.info_dict['device'])
            # node ids with label class i
            i_nid = nid[i_pos]
            # shuffle the i-th class node ids
            i_shuffle_ind = torch.randperm(len(i_pos)).to(self.info_dict['device'])
            i_nid_shuffled = i_nid[i_shuffle_ind]
            # get new id arrangement of the i-th class
            shuf_nid[i_pos] = i_nid_shuffled.to(self.info_dict['device'])

        return shuf_nid

    def gen_neg_nids(self):
        num_nodes = self.g.num_nodes
        nid = torch.arange(num_nodes, device=self.info_dict['device'])
        labels = self.pred_labels

        # randomly sample an instance as the negative sample, which is from a (estimated) different class
        shuf_nid = torch.randperm(num_nodes).to(self.info_dict['device'])
        for i in range(self.info_dict['out_dim']):
            sample_prob = 1 / len(nid) * torch.ones_like(nid)
            # position index of the i-th class
            i_pos = torch.where(labels == i)[0]
            # set the sampling prob to be 0 so that the node from the same class will not be sampled
            sample_prob[i_pos] = 0
            i_neg = torch.multinomial(sample_prob, len(i_pos), replacement=True).to(self.info_dict['device'])
            shuf_nid[i_pos] = i_neg

        return shuf_nid