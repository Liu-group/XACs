import os
import collections
import time
import multiprocessing
from copy import deepcopy
from argparse import Namespace
from typing import List, Optional
import numpy as np
from rdkit.Chem import MolFromSmiles
import torch
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from explain.GradCAM import GraphLayerGradCam
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
    smiles_train_init = data.smiles_train
    smiles_test = data.smiles_test
    y_train_init = data.y_train
    y_test = data.y_test
    cliff_mols_test = data.cliff_mols_test
    # further split data_train into train and val
    smiles_train, smiles_val, y_train, y_val = train_test_split(smiles_train_init, y_train_init, test_size=args.split[1]/(1.0-args.split[2]), shuffle=True, random_state=args.seed)
        
    fps_train = np.vstack([featurize_ecfp4(MolFromSmiles(sm)) for sm in smiles_train])
    fps_val = np.vstack([featurize_ecfp4(MolFromSmiles(sm)) for sm in smiles_val])
    fps_test = np.vstack([featurize_ecfp4(MolFromSmiles(sm)) for sm in smiles_test])
    metric_func = get_metric_func(metric=args.metric)
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
                 data: MoleculeDataset, 
                 ) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.
    :param model: Model to train.
    :param tr_fold: Training fold.
    :param val_fold: Validation fold.
    :param args: Arguments.
    :return: A list of ensemble scores.
    """
    data_train_init, data_test, batched_train_init = data.data_train, data.data_test, data.batched_data_train
    # further split data_train into train and val
    data_train, data_val = train_test_split(data_train_init, test_size=args.split[1]/(1.0-args.split[2]), shuffle=True, random_state=args.seed)
    batched_train, batched_val = train_test_split(batched_train_init, test_size=args.split[1]/(1.0-args.split[2]), shuffle=True, random_state=args.seed)

    train_loader = DataLoader(data_train, batch_size = args.batch_size, shuffle=False)
    val_loader = DataLoader(data_val, batch_size = args.batch_size, shuffle=False)

    batched_train_loader = DataLoader(batched_train, batch_size = args.batch_size, shuffle=False)
    batched_val_loader = DataLoader(batched_val, batch_size = args.batch_size, shuffle=False)

    loss_func = torch.nn.MSELoss()
    metric_func = get_metric_func(metric=args.metric)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5, min_lr=args.lr/20.0)

    best_score = float('inf') if args.minimize_score else -float('inf')

    for epoch in range(args.epochs):
        s_time = time.time()
        train_losses = train(args, model, train_loader, batched_train_loader, loss_func, optimizer)
        t_time = time.time() - s_time
        s_time = time.time()
        val_score, val_losses = evaluate(args, model, val_loader, batched_val_loader, loss_func, metric_func)
        v_time = time.time() - s_time
        scheduler.step(val_losses['val'][0])
        print('Epoch: {:04d}'.format(epoch),
                'loss_train: {:.6f}'.format(train_losses['train'][0]),
                'loss_val: {:.6f}'.format(val_losses['val'][0]),
                'pred_train: {:.6f}'.format(train_losses['pred'][0]),
                'pred_val: {:.6f}'.format(val_losses['pred'][0]),
                #'att_train: {:.6f}'.format(train_losses['att'][0]),
                #'att_val: {:.6f}'.format(val_losses['att'][0]),
                'direction_train: {:.6f}'.format(train_losses['direction'][0]),
                'direction_val: {:.6f}'.format(val_losses['direction'][0]),
                'sparsity_train: {:.6f}'.format(train_losses['sparsity'][0]),
                'sparsity_val: {:.6f}'.format(val_losses['sparsity'][0]),
                '{:.4s}_val: {:.4f}'.format(args.metric, val_score),
                # 'auc_val: {:.4f}'.format(avg_val_score),
                'cur_lr: {:.5f}'.format(optimizer.param_groups[0]['lr']),
                't_time: {:.4f}s'.format(t_time),
                'v_time: {:.4f}s'.format(v_time))
        if args.opt_goal != 'MSE':
            val_score = val_losses['val'][0]
            metric = 'loss'
        else:
            metric = args.metric
        if args.minimize_score and val_score < best_score or \
                not args.minimize_score and val_score > best_score:
            best_score, best_epoch = val_score, epoch
            best_model = deepcopy(model)        
            if args.save_checkpoints:
                if args.checkpoint_path is None:
                    args.checkpoint_path = os.path.join(args.save_dir, f'{args.loss}_model.pt')
                save_checkpoint(args.checkpoint_path, model, args)                
                print('saved checkpoint')
        if epoch - best_epoch > args.early_stop_epoch:
            break
    print('best epoch: {:04d}'.format(best_epoch))
    print('best val {:.4s}: {:.4f}'.format(metric, best_score))
    
    if data_test is not None:
        test_loader = DataLoader(data_test, batch_size = args.batch_size, shuffle=False)
        #model = load_checkpoint(args)       
        test_score, test_cliff_score  = predict(args, best_model, test_loader, loss_func, metric_func)

        return best_model, test_score, test_cliff_score
        

def train(args, model, train_loader, batched_train_loader, loss_func, optimizer):
    """
    Trains a model for an epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train().to(device)
    losses = collections.defaultdict(list)
    total_loss, pred_loss, att_loss, direct_loss, sparsity_loss = 0.0, 0.0, 0.0, 0.0, 0.0    
    graph_count = 0
    att_loss_weight = float(args.att_loss_weight)
    sparsity_loss_weight = float(args.sparsity_loss_weight)
    direction_loss_weight = float(args.direction_loss_weight)

    for data, batched_data in zip(train_loader, batched_train_loader):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        edge_attr = data.edge_attr.to(device)
        batch = data.batch.to(device)
        y = data.target.to(device)
        #################
        if 'att' in args.loss:
            att_label = data.att_label.to(device) 
            nan_mask = torch.isnan(att_label)
            att_trans = torch.abs(att)
            att_trans = torch.tanh(att_trans)
            if torch.all(nan_mask):
                attribution_prior = torch.tensor(0.0, requires_grad=True)
            else:
                attribution_prior = loss_func(att_trans[~nan_mask], att_label[[~nan_mask]])
        ####################
        # this will determine whether cliff pair will be used in training
        if args.show_direction_loss:
            batched_x, batched_edge_attr = batched_data.x.to(device), batched_data.edge_attr.to(device)
            batched_edge_index = batched_data.edge_index.to(device)
            pred_mask = batched_data.pred_mask.to(device)
            batched_batch = batched_data.batch.to(device)
            condition = pred_mask > 0.
            row_cond = condition.all(1)           
            ### attribution ####
            model.eval()
            explain_method = GraphLayerGradCam(model, model.convs[-1])
            att = explain_method.attribute((batched_x, batched_edge_attr), additional_forward_args=(batched_edge_index, batched_batch), return_gradients=args.return_gradients)
            att = att.reshape(-1, 1)
            ### sparsity ####
            sparsity_prior = torch.norm(att, p=args.norm)
            sparsity_loss += sparsity_prior.item()
            ####################
            masked_att = att*batched_data.atom_mask.to(device)
            masked_att = scatter(masked_att, batched_batch, dim=0, reduce='add') 
            direction_prior = 0.0
            direction_prior += pairwise_ranking_loss(masked_att, batched_data.potency_diff.reshape(-1, batched_data.atom_mask.shape[1]).to(device))
            direct_loss += direction_prior.item()
        ####################
        model.train()
        out = model(x=x, edge_attr=edge_attr, edge_index=edge_index, batch=batch)

        loss = loss_func(out.reshape(-1, 1), y.reshape(-1, 1))
        if args.loss == 'MSE':
            train_loss = loss 
        elif args.loss == 'MSE+att':
            train_loss = loss + att_loss_weight*attribution_prior
        elif args.loss == 'MSE+att+sparsity':
            train_loss = loss + att_loss_weight*attribution_prior + sparsity_loss_weight*sparsity_prior
        elif args.loss == 'MSE+sparsity':
            train_loss = loss + sparsity_loss_weight*sparsity_prior
        elif args.loss == 'MSE+direction+sparsity':
            train_loss = loss + direction_loss_weight*direction_prior + sparsity_loss_weight*sparsity_prior
        elif args.loss == 'MSE+direction':  
            train_loss = loss + direction_loss_weight*direction_prior
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_loss += train_loss.item()
        pred_loss += loss.item()
        #att_loss +=  attribution_prior.item()
        graph_count += data.num_graphs
    
    losses['train'].append(total_loss/graph_count)
    losses['pred'].append(pred_loss/graph_count)
    #losses['att'].append(att_loss/graph_count)
    losses['direction'].append(direct_loss/graph_count)
    losses['sparsity'].append(sparsity_loss/graph_count)

    return losses

def evaluate(args, model, val_loader, batched_val_loader, loss_func, metric_func):
    """
    Evaluates a model on a dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    losses = collections.defaultdict(list)
    total_loss, pred_loss, att_loss, direct_loss, sparsity_loss = 0.0, 0.0, 0.0, 0.0, 0.0
    att_loss_weight, sparsity_loss_weight, direction_loss_weight = float(args.att_loss_weight), float(args.sparsity_loss_weight), float(args.direction_loss_weight)
    y_pred, y_true = [], []
    graph_count = 0
    with torch.no_grad():
        for data, batched_data in zip(val_loader, batched_val_loader):
            x, edge_index = data.x.to(device), data.edge_index.type(torch.LongTensor).to(device)
            edge_attr = data.edge_attr.to(device)
            batch = data.batch.to(device)
            y = data.target.to(device)
            
            #################
            if 'att' in args.loss:
                att_label = data.att_label.to(device)
                nan_mask = torch.isnan(att_label)
                att_trans = torch.abs(att)
                att_trans = torch.tanh(att_trans)
                if torch.all(nan_mask):
                    attribution_prior = torch.tensor(0.0, requires_grad=True)
                else:
                    attribution_prior = loss_func(att_trans[~nan_mask], att_label[[~nan_mask]])
            ####################
            if args.show_direction_loss:
                batched_x, batched_edge_attr = batched_data.x.to(device), batched_data.edge_attr.to(device)
                batched_edge_index = batched_data.edge_index.to(device)
                pred_mask = batched_data.pred_mask.to(device)
                batched_batch = batched_data.batch.to(device)
                
                ### attribution ####
                explain_method = GraphLayerGradCam(model, model.convs[-1])
                att = explain_method.attribute((batched_x, batched_edge_attr), additional_forward_args=(batched_edge_index, batched_batch), return_gradients=args.return_gradients)
                att = att.reshape(-1, 1)
                ####################            
                sparsity_prior = torch.norm(att, p=args.norm)
                sparsity_loss += sparsity_prior.item()
                ####################   
                masked_att = att*batched_data.atom_mask.to(device)
                masked_att = scatter(masked_att, batched_batch, dim=0, reduce='add') 
                direction_prior = 0.0
                direction_prior += pairwise_ranking_loss(masked_att, batched_data.potency_diff.reshape(-1, batched_data.atom_mask.shape[1]).to(device))
                direct_loss += direction_prior.item()
            out = model(x=x, edge_attr=edge_attr, edge_index=edge_index, batch=batch)
            ####################
            loss = loss_func(out.reshape(-1, 1), y.reshape(-1, 1))
            if args.loss == 'MSE':
                val_loss = loss 
            elif args.loss == 'MSE+att':
                val_loss = loss + att_loss_weight*attribution_prior
            elif args.loss == 'MSE+att+sparsity':
                val_loss = loss + att_loss_weight*attribution_prior + args.sparsity_loss_weight*sparsity_prior
            elif args.loss == 'MSE+sparsity':
                val_loss = loss + sparsity_loss_weight*sparsity_prior
            elif args.loss == 'MSE+direction+sparsity':
                val_loss = loss + direction_loss_weight*direction_prior + args.sparsity_loss_weight*sparsity_prior
            elif args.loss == 'MSE+direction':  
                val_loss = loss + direction_loss_weight*direction_prior
            #att_loss +=  attribution_prior.item()
            total_loss += val_loss.item()
            graph_count += data.num_graphs
            pred_loss += loss.item()
            y_pred += list(out.cpu().detach().reshape(-1))
            y_true += list(y.cpu().detach().reshape(-1))
    val_score = metric_func(y_true, y_pred)
    
    losses['val'].append(total_loss/graph_count)
    losses['pred'].append(pred_loss/graph_count)
    #losses['att'].append(att_loss/graph_count)
    losses['direction'].append(direct_loss/graph_count)
    losses['sparsity'].append(sparsity_loss/graph_count)

    return val_score, losses

def predict(args, model, test_loader, loss_func, metric_func):
    """
    Evaluates a model on a test dataset.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    y_pred, y_true, cliffs = [], [], []
    test_batch_loss =  0.0
    for data in test_loader:
        x, edge_index = data.x.to(device), data.edge_index.type(torch.LongTensor).to(device)
        edge_attr = data.edge_attr.to(device)
        batch = data.batch.to(device)
        cliffs += data.cliff.data.cpu().tolist()
        y = data.target.to(device)
        out = model(x, edge_attr, edge_index, batch)

        #indices_list = get_batch_indices(batch.cpu().tolist())
        explain_method = GraphLayerGradCam(model, model.convs[-1])
        att = explain_method.attribute((x, edge_attr), additional_forward_args=(edge_index, batch))
        att = att.cpu().detach().reshape(-1, 1)

        loss = loss_func(out.reshape(-1, 1), y.reshape(-1, 1))
        test_batch_loss += loss.item()
        y_pred += list(out.cpu().detach().reshape(-1))
        y_true += list(y.cpu().detach().reshape(-1))
    test_score = metric_func(y_true, y_pred)
    y_pred_cliff = [y_pred[i] for i in range(len(y_pred)) if cliffs[i]==1]
    y_true_cliff = [y_true[i] for i in range(len(y_true)) if cliffs[i]==1]
    test_cliff_score = metric_func(y_true_cliff, y_pred_cliff)
    print('test {:.4s}: {:.4f}'.format(args.metric, test_score))
    print('test cliff {:.4s}: {:.4f}'.format(args.metric, test_cliff_score))
    
    return test_score, test_cliff_score

