import os
import numpy as np
import collections
import torch
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict
from XACs.utils.utils import makedirs, set_seed, load_checkpoint, get_deg
from XACs.dataset import MoleculeDataset, pack_data
from XACs.models.GNN import GNN
from XACs.train import run_training
from XACs.evaluate import evaluate_gnn_explain_direction, evaluate_rf_explain_direction, run_evaluation
from copy import deepcopy

def cross_validate(args, dataset: MoleculeDataset):
    init_seed = args.seed
    # Run training with different random seeds for each fold
    all_scores = collections.defaultdict(list)
    for fold_num in range(args.num_folds):
        print(f'Fold {fold_num}')
        current_args = deepcopy(args)
        current_args.seed = init_seed + fold_num
        set_seed(seed=current_args.seed)
        data_train, data_val, data_test = dataset.split_data(split_ratio=current_args.split, 
                                                            split_method=current_args.split_method, 
                                                            seed=args.seed, 
                                                            save_split=True)
        if current_args.conv_name == 'pna':
            current_args.deg = get_deg(data_train)
        if current_args.loss != 'MSE':
            data_train = pack_data(data_train, dataset.cliff_dict)
        model = GNN(num_node_features=args.num_node_features, 
                    num_edge_features=args.num_edge_features,
                    node_hidden_dim=args.node_hidden_dim,
                    edge_hidden_dim=args.edge_hidden_dim,
                    num_classes=args.num_classes,
                    conv_name=args.conv_name,
                    num_layers=args.num_layers,
                    hidden_dim=args.hidden_dim,
                    dropout_rate=args.dropout_rate,
                    pool=args.pool,
                    heads=args.heads,
                    uncom_pool=args.uncom_pool,
                    embed_method=args.embed_method,
                    deg=current_args.deg,
                    )
        # get the number of parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of trainable params: ", total_params)    
        best_val_scores = run_training(current_args, model, data_train, data_val)
        # For each metric, load the corresponding best model and evaluate
        for i, metric in enumerate(args.metric):
            print(f"\nEvaluating best model for {metric}:")
            # Load the best model for this metric
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

