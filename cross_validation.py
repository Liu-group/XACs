import os
import numpy as np
import collections
import torch
from typing import List, Dict
from torch_geometric.loader import DataLoader
from XACs.utils.utils import makedirs, set_seed, load_checkpoint, get_deg
from XACs.utils.metrics import get_metric_func
from XACs.dataset import MoleculeDataset, pack_data
from XACs.models.GNN import GNN
from XACs.train import run_training, run_cv_train, predict
from XACs.evaluate import evaluate_gnn_explain_direction, evaluate_gnn_explain_direction_ensemble, evaluate_rf_explain_direction, run_evaluation
from copy import deepcopy

def cross_validate(args, dataset: MoleculeDataset):
    init_seed = args.seed
    # Run training with different random seeds for each fold
    all_scores = collections.defaultdict(list)
    
    # Create a GNN config dictionary once
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
    
    for fold_num in range(args.num_folds):
        print(f'Fold {fold_num}')
        current_args = deepcopy(args)
        current_args.seed = init_seed + fold_num
        
        if current_args.ensemble:
            data_train, data_test = dataset.split_data(split_ratio=current_args.split, 
                                                      split_method=current_args.split_method, 
                                                      seed=current_args.seed, 
                                                      save_split=True)
            if current_args.loss != 'MSE':
                data_train = pack_data(data_train, dataset.cliff_dict)            
            run_cv_train(current_args, data_train, gnn_config)
            metric_funcs = {metric: get_metric_func(metric=metric) for metric in args.metric}
            for i, (metric, func) in enumerate(metric_funcs.items()):
                print(f"\nEvaluating best ensemble model for {metric}:")
                models = []
                for j in range(5):
                    current_args.checkpoint_path = os.path.join(args.model_dir, args.dataset, 
                                                        f'{args.dataset}_{args.loss}_model_{(init_seed + fold_num) * 10 + j}_{metric}.pt')
                    if current_args.save_checkpoints and os.path.exists(current_args.checkpoint_path):
                        best_model = load_checkpoint(current_args)
                    else:
                        print(f"Warning: Checkpoint for {metric} not found, using last model")
                        best_model = models[j]
                    models.append(best_model)
                if i == 0:
                    gnn_score, _ = evaluate_gnn_explain_direction_ensemble(current_args, dataset, data_test, models)
                    for key, value in gnn_score.items():
                        all_scores[key].append(value)
                    data_test = pack_data(data_test, dataset.cliff_dict, space=dataset.data_all)    
                y_preds, y_trues = [], []
                test_loader = DataLoader(data_test, batch_size = 1, shuffle=False)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                loss_func = torch.nn.MSELoss() if current_args.task == 'regression' else torch.nn.BCEWithLogitsLoss()
                for j in range(5):
                    model = models[j]
                    model.to(device)
                    y_pred, y_true, cliffs = predict(current_args, model, test_loader, loss_func, device)
                    y_preds.append(y_pred)
                    y_trues.append(y_true)
                y_preds = torch.cat(y_preds, dim=1)
                y_trues = torch.cat(y_trues, dim=1)
                y_ensemble_pred = torch.mean(y_preds, dim=1)
                y_ensemble_true = torch.mean(y_trues, dim=1)
                cliffs = np.array([cliffs[i].item() for i in range(len(cliffs))])
                y_ensemble_pred_cliff = y_ensemble_pred[cliffs==1]
                y_ensemble_true_cliff = y_ensemble_true[cliffs==1]
                test_scores = func(y_ensemble_true, y_ensemble_pred)
                print('test {:.4s}: {:.3f}'.format(metric, test_scores))
                test_cliff_scores = func(y_ensemble_true_cliff, y_ensemble_pred_cliff)
                print('test cliff {:.4s}: {:.3f}'.format(metric, test_cliff_scores))        
                all_scores[f'gnn_test_{metric}'].append(test_scores)
                all_scores[f'gnn_test_cliff_{metric}'].append(test_cliff_scores)
        else:
            set_seed(seed=current_args.seed)
            data_train, data_val, data_test = dataset.split_data(split_ratio=current_args.split, 
                                                                split_method=current_args.split_method, 
                                                                seed=current_args.seed, 
                                                                save_split=True)
            if current_args.loss != 'MSE':
                data_train = pack_data(data_train, dataset.cliff_dict)
            if current_args.conv_name == 'pna':
                current_args.deg = get_deg(data_train)
                gnn_config['deg'] = current_args.deg
            model = GNN(**gnn_config)
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("Total number of trainable params: ", total_params)    
            run_training(current_args, model, data_train, data_val)
            for i, metric in enumerate(args.metric):
                print(f"\nEvaluating best model for {metric}:")
                current_args.checkpoint_path = os.path.join(args.model_dir, args.dataset, 
                                                        f'{args.dataset}_{args.loss}_model_{current_args.seed}_{metric}.pt')
                if current_args.save_checkpoints and os.path.exists(current_args.checkpoint_path):
                    best_model = load_checkpoint(current_args)
                else:
                    print(f"Warning: Checkpoint for {metric} not found, using last model")
                    best_model = model
                if i == 0:
                    # Evaluate explanation direction
                    gnn_score, _ = evaluate_gnn_explain_direction(dataset, data_test, best_model)
                    for key, value in gnn_score.items():
                        all_scores[f"{key}_{metric}"].append(value)
                    # Prepare test data
                    data_test = pack_data(data_test, dataset.cliff_dict, space=dataset.data_all)
            
                # Evaluate on test set
                test_scores, test_cliff_scores, explan_acc = run_evaluation(current_args, best_model, data_test)
                
                # Store only the score for the current metric
                if metric in test_scores:
                    all_scores[f'gnn_test_{metric}_best'].append(test_scores[metric])
                
                if metric in test_cliff_scores:
                    all_scores[f'gnn_test_cliff_{metric}_best'].append(test_cliff_scores[metric])
            
                all_scores[f'gnn_explanation_accuracy_{metric}'].append(explan_acc)
            
                # Clean up to free memory
                if current_args.save_checkpoints:
                    del best_model
                    torch.cuda.empty_cache()
                
    # Report scores for each fold
    print(f'\n{args.num_folds}-fold cross validation results:')

    for key, fold_scores in all_scores.items():
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f'{args.dataset} ==> {key} = {mean_score:.3f} +/- {std_score:.3f}')
        if args.show_individual_scores:
            for fold_num, scores in enumerate(fold_scores):
                print(f'Seed {init_seed + fold_num} ==> {key} = {scores:.3f}')

    print("args:", args)
    
    # Return the primary metric's mean and std
    primary_metric = args.metric[0]
    primary_key = f'gnn_test_{primary_metric}_best'
    return np.mean(all_scores[primary_key]), np.std(all_scores[primary_key])

