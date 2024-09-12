import os
import collections
import time
import multiprocessing
from copy import deepcopy
from argparse import Namespace
from typing import List, Optional
import numpy as np
import gc
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_scatter import scatter
from XACs.utils.explain_utils import process_layer_gradients_and_eval
from XACs.utils.utils import save_checkpoint, load_checkpoint, pairwise_ranking_loss
from XACs.utils.metrics import get_metric_func
from XACs.dataset import MoleculeDataset
from XACs.GNN import GNN
from sklearn.model_selection import train_test_split

def run_training(args: Namespace,
                 model: GNN, 
                 data_train: List[Data], 
                 data_val: List[Data],
                 ) -> List[float]:
    """
    Trains a model and returns the model with the highest validation score.
    :param model: Model to train.
    :param data_train: Training data.
    :param data_val: Validation data.
    :return: Model with the highest validation score.
    """
    train_loader = DataLoader(data_train, batch_size = args.batch_size, shuffle=False)
    val_loader = DataLoader(data_val, batch_size = args.batch_size, shuffle=False)

    loss_func = torch.nn.MSELoss() if args.task == 'regression' else torch.nn.BCEWithLogitsLoss()
    metric = args.metric
    metric_func = get_metric_func(metric=metric)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, min_lr=args.min_lr)

    best_score = float('inf') if args.minimize_score else -float('inf')
    losses = collections.defaultdict(list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(args.epochs):
        s_time = time.time()
        train_losses = train(args, epoch, model, train_loader, loss_func, optimizer, device)
        t_time = time.time() - s_time
        s_time = time.time()
        val_score, val_loss = evaluate(args, model, val_loader, loss_func, metric_func, device)
        v_time = time.time() - s_time
        scheduler.step(val_score)

        #losses['train'].append(train_losses['train'][0])
        losses['train_pred'].append(train_losses['pred'][0])
        losses['train_explanation'].append(train_losses['explanation'][0])
        losses['val'].append(val_loss)

        print('Epoch: {:04d}'.format(epoch),
                'train_pred_loss: {:.6f}'.format(train_losses['pred'][0]),
                'train_xloss: {:.6f}'.format(train_losses['explanation'][0]),
                'val_pred_loss: {:.6f}'.format(val_loss),
                '{:.4s}_val: {:.4f}'.format(args.metric, val_score),
                'cur_lr: {:.5f}'.format(optimizer.param_groups[0]['lr']),
                't_time: {:.4f}s'.format(t_time),
                'v_time: {:.4f}s'.format(v_time))
        # Save model checkpoint if improved validation score
        if (args.minimize_score and val_score < best_score) or \
                (not args.minimize_score and val_score > best_score):
            best_score, best_epoch = val_score, epoch  
            if args.save_checkpoints == True: 
                os.makedirs(os.path.join(args.model_dir, args.dataset), exist_ok=True)
                args.checkpoint_path = os.path.join(args.model_dir, args.dataset, f'{args.dataset}_{args.loss}_model_{args.seed}.pt')
                save_checkpoint(args.checkpoint_path, model, args)              
        if args.early_stop_epoch != None and epoch - best_epoch > args.early_stop_epoch:
            break
    print('best epoch: {:04d}'.format(best_epoch))
    print('best val {:.4s}: {:.4f}'.format(metric, best_score))
    return best_score


def train(args, epoch, model, train_loader, loss_func, optimizer, device):
    """
    Trains a model for an epoch.
    """
    model.train()
    losses = collections.defaultdict(list)
    total_loss, pred_loss, explanation_loss, weighted_explanation_loss = 0.0, 0.0, 0.0, 0.0 
    graph_count, num_explanation = 0, 0
    com_loss_weight, uncom_loss_weight = float(args.com_loss_weight), float(args.uncom_loss_weight)
    len_dataloader = len(train_loader)
    for i, data in enumerate(train_loader):
        data.to(device)
        target = data.target.reshape(-1, 1).double().to(device)
        if args.loss == 'MSE':
            output = model(data.x, data.edge_attr, data.edge_index.type(torch.LongTensor).to(device), data.batch.to(device))
            common_prior = uncom_prior = 0.
        elif args.gnes:
            output, att = model.gnes_forward(data.x, data.edge_attr, data.edge_index.type(torch.LongTensor).to(device), data.batch.to(device))
            uncom_loss_weight = uncom_prior = 0.0
            common_prior = torch.abs(att).sum()
            explanation_loss += common_prior.item()
            weighted_explanation_loss += com_loss_weight*common_prior.item()
            num_explanation += data.num_graphs
        else:
            if args.xscheduler:
                p = float(i + epoch * len_dataloader) / args.epochs / len_dataloader
                alpha = 2. / (1. + np.exp(-10 * p)) - 1
                com_loss_weight = com_loss_weight * alpha
                uncom_loss_weight = uncom_loss_weight * alpha
            potency_diff = data.potency_diff.to(device)
            output, pooled_uncom_att, common_att = model.explanation_forward(data)
            uncom_prior = pairwise_ranking_loss(pooled_uncom_att, potency_diff)
            common_prior = torch.square(common_att).sum()
            explanation_loss += uncom_prior.item() + common_prior.item()
            weighted_explanation_loss += com_loss_weight*common_prior.item() + uncom_loss_weight*uncom_prior.item()
            num_explanation += torch.count_nonzero(potency_diff).item()
        loss = loss_func(output, target)
        train_loss = loss + com_loss_weight*common_prior + uncom_loss_weight*uncom_prior
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        pred_loss += loss.item()*data.num_graphs
        graph_count += data.num_graphs
    losses['pred'].append(pred_loss/graph_count)
    losses['explanation'].append(explanation_loss/num_explanation if num_explanation > 0 else 0.0)
    losses['weighted_explanation'].append(weighted_explanation_loss/num_explanation if num_explanation > 0 else 0.0)

    return losses

def evaluate(args, model, val_loader, loss_func, metric_func, device):
    model.eval()
    losses = collections.defaultdict(list)
    total_loss, pred_loss = 0.0, 0.0
    y_pred, y_true = torch.zeros(0, args.num_classes), torch.zeros(0, 1)
    graph_count = 0
    with torch.no_grad():
        for data in val_loader:
            x, edge_index = data.x.to(device), data.edge_index.type(torch.LongTensor).to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            target = data.target.reshape(-1, 1).double().to(device)
            out = model(x, edge_attr, edge_index, batch)
            loss = loss_func(out, target)
            total_loss += loss.item()*data.num_graphs
            graph_count += data.num_graphs
            y_pred = torch.cat((y_pred, out.cpu().detach().reshape(-1, args.num_classes)))
            y_true = torch.cat((y_true, target.cpu().detach()))
    val_score = metric_func(y_true, torch.sigmoid(y_pred) if args.task == 'classification' else y_pred)

    return val_score, total_loss/graph_count

def predict(args, model, test_loader, loss_func, metric_func, device):
    model.eval()
    y_pred, y_true, cliffs = torch.zeros(0, args.num_classes), torch.zeros(0, 1), torch.zeros(0, 1)
    total_loss, explanation_loss, weighted_explanation_loss =  0.0, 0.0, 0.0
    graph_count, num_explanation, num_true_explanation = 0, 0, 0
    com_loss_weight, uncom_loss_weight = float(args.com_loss_weight), float(args.uncom_loss_weight)
    for data in test_loader:
        data.to(device)
        target = data.target.reshape(-1, 1).double().to(device)
        cliffs = torch.cat((cliffs, data.cliff.cpu().reshape(-1, 1)))
        potency_diff = data.potency_diff.to(device)
        output, pooled_uncom_att_diff, common_att = model.explanation_forward(data)
        uncom_prior = pairwise_ranking_loss(pooled_uncom_att_diff, potency_diff)
        num_true_explanation += ((pooled_uncom_att_diff * potency_diff) > 0).sum().item()
        common_prior = torch.square(common_att).sum()
        loss = loss_func(output, target)

        explanation_loss += uncom_prior.item() + common_prior.item()
        weighted_explanation_loss += com_loss_weight*common_prior.item() + uncom_loss_weight*uncom_prior.item()      

        num_explanation += torch.count_nonzero(potency_diff).item()
        total_loss += loss.item()*data.num_graphs
        y_pred = torch.cat((y_pred, output.cpu().detach().reshape(-1, args.num_classes)))
        y_true = torch.cat((y_true, target.cpu().detach()))

    test_score = metric_func(y_true, torch.sigmoid(y_pred) if args.task == 'classification' else y_pred)
    print('test {:.4s}: {:.3f}'.format(args.metric, test_score))
    test_cliff_score, explan_acc = 0, 0
    if num_explanation > 0:
        y_pred_cliff = y_pred[cliffs==1]
        y_true_cliff = y_true[cliffs==1]
        if sum(y_true_cliff) == 0 or sum(y_true_cliff) == len(y_true_cliff):
            test_cliff_score = 0
        else:
            test_cliff_score = metric_func(y_true_cliff, y_pred_cliff)
            print('test cliff {:.4s}: {:.3f}'.format(args.metric, test_cliff_score))
        # explanation accuracy
        explan_acc = num_true_explanation/num_explanation
        print('Total number of explanations: {}'.format(num_explanation))
        print('explanation accuracy: {:.3f}'.format(explan_acc))
        if args.task == 'regression':
            # get r2
            r2_metric = get_metric_func(metric='r2')
            r2_test = r2_metric(y_true, y_pred)
            print('test r2: {:.3f}'.format(r2_test))
            r2_cliff_test = r2_metric(y_true_cliff, y_pred_cliff)
            print('test cliff r2: {:.3f}'.format(r2_cliff_test))            
    print('explanation loss: {:.3f}'.format(explanation_loss))
    print('weighted explanation loss: {:.3f}'.format(weighted_explanation_loss))
    return test_score, test_cliff_score, explan_acc
