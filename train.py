import os
import collections
import time
import multiprocessing
from copy import deepcopy
from argparse import Namespace
from typing import List, Optional
import numpy as np
from rdkit.Chem import MolFromSmiles
import gc
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_scatter import scatter
from explain.explain_utils import process_layer_gradients_and_eval
from utils import save_checkpoint, load_checkpoint
from metrics import get_metric_func
from dataset import MoleculeDataset
from GNN import GNN
from utils import pairwise_ranking_loss, get_batch_indices
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from rf_utils import featurize_ecfp4

N_TREES_LIST = [100, 250, 500, 1000]
N_JOBS = int(os.getenv("LSB_DJOB_NUMPROC", multiprocessing.cpu_count()))
print("Number of Jobs: ", N_JOBS)

def train_test_rf(args: Namespace, data: MoleculeDataset):
    smiles_train = [data.data_train[i].smiles for i in range(len(data.data_train))]
    smiles_val = [data.data_val[i].smiles for i in range(len(data.data_val))]
    smiles_test = [data.data_test[i].smiles for i in range(len(data.data_test))]

    y_train = [data.data_train[i].target for i in range(len(data.data_train))]
    y_val = [data.data_val[i].target for i in range(len(data.data_val))]
    y_test = [data.data_test[i].target for i in range(len(data.data_test))]

    cliff_mols_test = [data.data_test[i].cliff for i in range(len(data.data_test))]
    
    fps_train = np.vstack([featurize_ecfp4(MolFromSmiles(sm)) for sm in smiles_train])
    fps_val = np.vstack([featurize_ecfp4(MolFromSmiles(sm)) for sm in smiles_val]) 
    fps_test = np.vstack([featurize_ecfp4(MolFromSmiles(sm)) for sm in smiles_test])

    metric_func = get_metric_func(metric=args.metric)
    if fps_val is not None:
        best_score = float('inf') if args.minimize_score else -float('inf')
        for N_TREES in N_TREES_LIST:
            model_rf = RandomForestRegressor(n_estimators=N_TREES, n_jobs=N_JOBS)
            model_rf.fit(fps_train, y_train)
            y_val = model_rf.predict(fps_val)
            val_score = metric_func(y_val, y_val)
            if args.minimize_score and val_score < best_score or \
                    not args.minimize_score and val_score > best_score:
                best_score = val_score
                best_N_TREES = N_TREES
        print('best N_TREES: {:04d}'.format(best_N_TREES))
        print('best val {:.4s}: {:.6f}'.format(args.metric, best_score))
    else:
        best_N_TREES = 100
    model_rf = RandomForestRegressor(n_estimators=best_N_TREES, n_jobs=N_JOBS)
    model_rf.fit(fps_train, y_train)
    y_pred = model_rf.predict(fps_test)
    test_score = metric_func(y_test, y_pred)
    y_cliff_test = [y_test[i] for i in range(len(y_test)) if cliff_mols_test[i]==1]
    y_cliff_pred = [y_pred[i] for i in range(len(y_test)) if cliff_mols_test[i]==1]
    test_cliff_score = metric_func(y_cliff_test, y_cliff_pred)
    pcc_test = np.corrcoef((y_test, y_pred))[0, 1]
    print('rf test {:.4s}: {:.4f}'.format(args.metric, test_score))
    print('rf test cliff {:.4s}: {:.4f}'.format(args.metric, test_cliff_score))
    print('rf test pcc: {:.4f}'.format(pcc_test))
    return model_rf, test_score, test_cliff_score


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

    loss_func = torch.nn.MSELoss()
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
        train_losses = train(args, model, train_loader, loss_func, optimizer, device)
        t_time = time.time() - s_time
        s_time = time.time()
        val_score, val_loss = evaluate(args, model, val_loader, loss_func, metric_func, device)
        v_time = time.time() - s_time
        scheduler.step(val_score)

        losses['train'].append(train_losses['train'][0])
        losses['train_pred'].append(train_losses['pred'][0])
        losses['train_explanation'].append(train_losses['explanation'][0])
        losses['val'].append(val_loss)

        print('Epoch: {:04d}'.format(epoch),
                'loss_train: {:.6f}'.format(train_losses['train'][0]),
                'pred_train: {:.6f}'.format(train_losses['pred'][0]),
                'explanation_train: {:.6f}'.format(train_losses['explanation'][0]),
                'weighted_explanation_train: {:.6f}'.format(train_losses['weighted_explanation'][0]),
                'loss_val: {:.6f}'.format(val_loss),
                '{:.4s}_val: {:.4f}'.format(args.metric, val_score),
                'cur_lr: {:.5f}'.format(optimizer.param_groups[0]['lr']),
                't_time: {:.4f}s'.format(t_time),
                'v_time: {:.4f}s'.format(v_time))
        # Save model checkpoint if improved validation score
        if args.minimize_score and val_score < best_score or \
                not args.minimize_score and val_score > best_score:
            best_score, best_epoch = val_score, epoch  
            if args.save_checkpoints == True: 
                if args.checkpoint_path is None:
                    args.checkpoint_path = os.path.join(args.save_dir, f'{args.dataset}_{args.loss}_model_{args.seed}_ablate_uncom.pt')
                save_checkpoint(args.checkpoint_path, model, args)              
        if args.early_stop_epoch != None and epoch - best_epoch > args.early_stop_epoch:
            break
    print('best epoch: {:04d}'.format(best_epoch))
    print('best val {:.4s}: {:.4f}'.format(metric, best_score))
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best_score


def train(args, model, train_loader, loss_func, optimizer, device):
    """
    Trains a model for an epoch.
    """
    model.train()
    losses = collections.defaultdict(list)
    total_loss, pred_loss, explanation_loss, weighted_explanation_loss = 0.0, 0.0, 0.0, 0.0 
    graph_count = 0
    com_loss_weight, uncom_loss_weight = float(args.com_loss_weight), float(args.uncom_loss_weight)
    ## New implementation ##
    for data in train_loader:
        data.to(device)
        target = data.target.reshape(-1, 1).to(device)
        if args.loss == 'MSE':
            output = model(data.x, data.edge_attr, data.edge_index.type(torch.LongTensor).to(device), data.batch.to(device))
            common_prior = uncom_prior = 0.
        else:
            potency_diff = data.potency_diff.to(device)
            output, sum_uncom_att, sum_common_att = model.explanation_forward(data)
            uncom_prior = pairwise_ranking_loss(sum_uncom_att, potency_diff)
            common_prior = torch.square(sum_common_att).sum()
            explanation_loss += uncom_prior.item() + common_prior.item()
            weighted_explanation_loss += com_loss_weight*common_prior.item() + uncom_loss_weight*uncom_prior.item()

        loss = loss_func(output.reshape(-1, 1), target)
        train_loss = loss + com_loss_weight*common_prior + uncom_loss_weight*uncom_prior
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item()*data.num_graphs
        pred_loss += loss.item()*data.num_graphs
        graph_count += data.num_graphs
    losses['train'].append(total_loss/graph_count)
    losses['pred'].append(pred_loss/graph_count)
    losses['explanation'].append(explanation_loss/graph_count)
    losses['weighted_explanation'].append(weighted_explanation_loss/graph_count)

    return losses

def evaluate(args, model, val_loader, loss_func, metric_func, device):
    model.eval()
    losses = collections.defaultdict(list)
    total_loss, pred_loss = 0.0, 0.0
    y_pred, y_true = [], []
    graph_count = 0
    with torch.no_grad():
        for data in val_loader:
            x, edge_index = data.x.to(device), data.edge_index.type(torch.LongTensor).to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            target = data.target.to(device)
            out = model(x, edge_attr, edge_index, batch)
            loss = loss_func(out.reshape(-1, 1), target.reshape(-1, 1))
            total_loss += loss.item()*data.num_graphs
            graph_count += data.num_graphs
            y_pred += list(out.cpu().detach().reshape(-1))
            y_true += list(target.cpu().detach().reshape(-1))
    val_score = metric_func(y_true, y_pred)

    return val_score, total_loss/graph_count

def predict(args, model, test_loader, loss_func, metric_func, device):
    model.eval()
    y_pred, y_true, cliffs = [], [], []
    total_loss, explanation_loss, weighted_explanation_loss =  0.0, 0.0, 0.0
    graph_count, num_explanation, num_true_explanation = 0, 0, 0
    com_loss_weight, uncom_loss_weight = float(args.com_loss_weight), float(args.uncom_loss_weight)
    for data in test_loader:
        data.to(device)
        
        target = data.target.reshape(-1, 1).to(device)
        cliffs += data.cliff.data.cpu().tolist()
        potency_diff = data.potency_diff.to(device)
        output, sum_uncom_att, common_att = model.explanation_forward(data)
        uncom_prior = pairwise_ranking_loss(sum_uncom_att, potency_diff)
        num_true_explanation += ((sum_uncom_att * potency_diff) > 0).sum().item()
        common_prior = torch.square(common_att).sum()
        loss = loss_func(output.reshape(-1, 1), target)

        explanation_loss += uncom_prior.item() + common_prior.item()
        weighted_explanation_loss += com_loss_weight*common_prior.item() + uncom_loss_weight*uncom_prior.item()      

        num_explanation += torch.count_nonzero(potency_diff).item()
        total_loss += loss.item()*data.num_graphs
        y_pred += list(output.cpu().detach().reshape(-1))
        y_true += list(target.cpu().detach().reshape(-1))

    test_score = metric_func(y_true, y_pred)
    y_pred_cliff = [y_pred[i] for i in range(len(y_pred)) if cliffs[i]==1]
    y_true_cliff = [y_true[i] for i in range(len(y_true)) if cliffs[i]==1]
    test_cliff_score = metric_func(y_true_cliff, y_pred_cliff)
    print('test {:.4s}: {:.4f}'.format(args.metric, test_score))
    print('test cliff {:.4s}: {:.4f}'.format(args.metric, test_cliff_score))
    # get r2
    r2_metric = get_metric_func(metric='r2')
    r2_test = r2_metric(y_true, y_pred)
    r2_cliff_test = r2_metric(y_true_cliff, y_pred_cliff)
    print('test r2: {:.4f}'.format(r2_test))
    print('test cliff r2: {:.4f}'.format(r2_cliff_test))
    # explanation accuracy
    explan_acc = num_true_explanation/num_explanation
    print('explanation accuracy: {:.4f}'.format(explan_acc))
    print('explanation loss: {:.4f}'.format(explanation_loss))
    print('weighted explanation loss: {:.4f}'.format(weighted_explanation_loss))
    return test_score, test_cliff_score, explan_acc
