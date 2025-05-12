import os
import collections
import time
import multiprocessing
from copy import deepcopy
from argparse import Namespace
from typing import List, Optional, Dict
import numpy as np
import gc
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_scatter import scatter
from XACs.utils.explain_utils import process_layer_gradients_and_eval
from XACs.utils.utils import save_checkpoint, load_checkpoint, pairwise_ranking_loss, get_deg
from XACs.utils.metrics import get_metric_func
from XACs.dataset import MoleculeDataset
from XACs.models.GNN import GNN
from sklearn.model_selection import train_test_split

def run_training(args: Namespace,
                 data_train: List[Data], 
                 data_val: List[Data],
                 ) -> Dict[str, float]:
    """
    :param model: Model to train.
    :param data_train: Training data.
    :param data_val: Validation data.
    :return: Dictionary of best validation scores for each metric.
    """
    gnn_config = {
                'num_node_features': args.num_node_features,
                'num_edge_features': args.num_edge_features,
                'node_hidden_dim': args.node_hidden_dim,
                'edge_hidden_dim': args.edge_hidden_dim,
                'num_classes': args.num_classes,
                'conv_name': args.conv_name,
                'num_layers': args.num_layers,
                'hidden_dim': args.hidden_dim,
                'dropout_rate': args.dropout_rate,
                'pool': args.pool,
                'heads': args.heads,
                'uncom_pool': args.uncom_pool,
                'embed_method': args.embed_method,
            }       
    if args.conv_name == 'pna':
        gnn_config['deg'] = get_deg(data_train)
    model = GNN(**gnn_config)
    train_loader = DataLoader(data_train, batch_size = args.batch_size, shuffle=False)
    val_loader = DataLoader(data_val, batch_size = args.batch_size, shuffle=False)

    loss_func = torch.nn.MSELoss() if args.task == 'regression' else torch.nn.BCEWithLogitsLoss()
    
    # Create dictionary of metric functions
    metric_funcs = {metric: get_metric_func(metric=metric) for metric in args.metric}
    
    # Track best scores for each metric
    best_scores = {}
    best_epochs = {}
    for metric in args.metric:
        best_scores[metric] = float('inf') if metric in ['rmse', 'mse', 'mae'] else -float('inf')
        best_epochs[metric] = 0
    
    # Also track best validation loss for early stopping
    best_val_loss = float('inf')
    best_val_loss_epoch = 0
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Use validation loss for LR scheduling
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=args.factor, patience=args.patience, min_lr=args.min_lr)

    losses = collections.defaultdict(list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(args.epochs):
        s_time = time.time()
        train_losses = train(args, epoch, model, train_loader, loss_func, optimizer, device)
        t_time = time.time() - s_time
        s_time = time.time()
        
        # Get all metrics
        val_scores, val_loss = evaluate(args, model, val_loader, loss_func, metric_funcs, device)
        
        v_time = time.time() - s_time
        # Use validation loss for scheduler
        scheduler.step(val_loss)

        losses['train_pred'].append(train_losses['pred'][0])
        losses['train_explanation'].append(train_losses['explanation'][0])
        losses['val'].append(val_loss)

        print('Epoch: {:04d}'.format(epoch),
                'train_pred_loss: {:.6f}'.format(train_losses['pred'][0]),
                'train_xloss: {:.6f}'.format(train_losses['explanation'][0]),
                'val_pred_loss: {:.6f}'.format(val_loss),
                'cur_lr: {:.5f}'.format(optimizer.param_groups[0]['lr']),
                't_time: {:.4f}s'.format(t_time),
                'v_time: {:.4f}s'.format(v_time))
                
        # Print all metrics
        for metric, score in val_scores.items():
            print('{:.4s}_val: {:.4f}'.format(metric, score))
        
        # Track best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            # Save a model based on validation loss
            if args.save_checkpoints:
                os.makedirs(os.path.join(args.model_dir, args.dataset), exist_ok=True)
                checkpoint_path = os.path.join(args.model_dir, args.dataset, 
                                             f'{args.dataset}_{args.loss}_model_{args.seed}_val_loss.pt')
                save_checkpoint(checkpoint_path, model, args)
                #print(f'Saved best validation loss model at epoch {epoch}')
                
        # Check for improvement in each metric and save corresponding model
        for metric, score in val_scores.items():
            minimize = metric in ['rmse', 'mse', 'mae']
            if (minimize and score < best_scores[metric]) or (not minimize and score > best_scores[metric]):
                best_scores[metric], best_epochs[metric] = score, epoch
                if args.save_checkpoints:
                    os.makedirs(os.path.join(args.model_dir, args.dataset), exist_ok=True)
                    checkpoint_path = os.path.join(args.model_dir, args.dataset, 
                                                 f'{args.dataset}_{args.loss}_model_{args.seed}_{metric}.pt')
                    save_checkpoint(checkpoint_path, model, args)
                    #print(f'Saved best {metric} model at epoch {epoch}')
                    
        # Early stopping based on validation loss
        if args.early_stop_epoch is not None and epoch - best_val_loss_epoch > args.early_stop_epoch:
            print(f'Early stopping triggered after {args.early_stop_epoch} epochs without improvement in validation loss')
            break
            
    # Print best results for each metric
    print('Best validation scores:')
    print('val_loss: {:.4f} at epoch {:04d}'.format(best_val_loss, best_val_loss_epoch))
    for metric, score in best_scores.items():
        print('{:.4s}_val: {:.4f} at epoch {:04d}'.format(metric, score, best_epochs[metric]))
    
    del model
    torch.cuda.empty_cache()
    return best_scores[args.metric[0]]
        

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
        target = data.target.reshape(-1, 1).to(device)
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

def evaluate(args, model, val_loader, loss_func, metric_funcs, device):
    """
    Evaluates a model on a validation set without performing backpropagation.
    """
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
            target = data.target.reshape(-1, 1).to(device)
            out = model(x, edge_attr, edge_index, batch)
            loss = loss_func(out, target)
            total_loss += loss.item()*data.num_graphs
            graph_count += data.num_graphs
            y_pred = torch.cat((y_pred, out.cpu().detach().reshape(-1, args.num_classes)))
            y_true = torch.cat((y_true, target.cpu().detach()))
    
    # Calculate all metrics
    val_scores = {}
    for metric, func in metric_funcs.items():
        val_scores[metric] = func(y_true, torch.sigmoid(y_pred) if args.task == 'classification' else y_pred)

    return val_scores, total_loss/graph_count

def predict(args, model, test_loader, loss_func, device):
    """
    Evaluates a model on a test set using explanation_forward (performing backpropagation).
    """
    model.eval()
    model.to(device)
    y_pred, y_true, cliffs = torch.zeros(0, args.num_classes), torch.zeros(0, 1), torch.zeros(0, 1)
    total_loss, explanation_loss, weighted_explanation_loss =  0.0, 0.0, 0.0
    graph_count, num_explanation, num_true_explanation = 0, 0, 0
    com_loss_weight, uncom_loss_weight = float(args.com_loss_weight), float(args.uncom_loss_weight)
    for data in test_loader:
        data.to(device)
        target = data.target.reshape(-1, 1).to(device)
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

    explan_acc = num_true_explanation/num_explanation
    print('Total number of explanations: {}'.format(num_explanation))
    print('explanation accuracy: {:.3f}'.format(explan_acc))
    #print('explanation loss: {:.3f}'.format(explanation_loss))
    #print('weighted explanation loss: {:.3f}'.format(weighted_explanation_loss))
    return y_pred, y_true, cliffs


def run_cv_train(args, data_train, gnn_config):
    """
    Trains a model on a K-Fold cross-validation set, returns the best model for each fold.
    Adopt from https://github.com/shenwanxiang/ACANet/blob/main/clsar/main.py#L185 _cv_split(
    """
    from sklearn.model_selection import StratifiedKFold
    gnn_config = {
                        'num_node_features': args.num_node_features,
                        'num_edge_features': args.num_edge_features,
                        'node_hidden_dim': args.node_hidden_dim,
                        'edge_hidden_dim': args.edge_hidden_dim,
                        'num_classes': args.num_classes,
                        'conv_name': args.conv_name,
                        'num_layers': args.num_layers,
                        'hidden_dim': args.hidden_dim,
                        'dropout_rate': args.dropout_rate,
                        'pool': args.pool,
                        'heads': args.heads,
                        'uncom_pool': args.uncom_pool,
                        'embed_method': args.embed_method,
                    }       
    if args.conv_name == 'pna':
        gnn_config['deg'] = get_deg(data_train)
    KFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    y_all = [data.target for data in data_train]
    cutoff = np.median(y_all)
    labels = [0 if i < cutoff else 1 for i in y_all]
    splits = [{'inner_train_idx': i, 'inner_val_idx': j} for i, j in KFold.split(labels, labels)]
    initial_fold_seed = args.seed
    for i, split in enumerate(splits):
        inner_train_data = [data_train[idx] for idx in split['inner_train_idx']]
        inner_val_data = [data_train[idx] for idx in split['inner_val_idx']]
        args.seed = initial_fold_seed * 10 + i
        if args.conv_name == 'pna':
            args.deg = get_deg(inner_train_data)
        gnn_config['deg'] = args.deg
        model = GNN(**gnn_config)    
        _ = run_training(args, model, inner_train_data, inner_val_data)
    print("All folds trained")
        
        
