import os
import numpy as np
import collections
import torch
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict
from XACs.utils.utils import makedirs, set_seed, load_checkpoint
from XACs.dataset import MoleculeDataset, pack_data
from XACs.GNN import GNN
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
                                                            seed=42, 
                                                            save_split=True)
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
                    ifp=args.ifp,
                    )
        # get the number of parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of trainable params: ", total_params)    
        best_val_score = run_training(current_args, model, data_train, data_val)
        best_model = load_checkpoint(current_args) if current_args.save_checkpoints else model
        
        gnn_score, _ = evaluate_gnn_explain_direction(dataset, data_test, best_model)
        for key, value in gnn_score.items():
            all_scores[key].append(value)

        data_test = pack_data(data_test, dataset.cliff_dict, space=dataset.data_all)
        test_score, test_cliff_score, explan_acc = run_evaluation(current_args, best_model, data_test)
        all_scores['gnn_test_score'].append(test_score)
        all_scores['gnn_test_cliff_score'].append(test_cliff_score)
        all_scores['gnn_explanation_accuracy'].append(explan_acc)

        ## not sure if necessary; the reset_parameters() function should also be checked.
        del best_model, model
        torch.cuda.empty_cache()
    # Report scores for each fold
    print(f'{args.num_folds}-fold cross validation')

    for key, fold_scores in all_scores.items():
        metric = '_' + args.metric if key=='gnn_test_score' or key=='gnn_test_cliff_score' else ''        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f'{args.dataset} ==> {key}{metric} = {mean_score:.3f} +/- {std_score:.3f}')
        if args.show_individual_scores:
            for fold_num, scores in enumerate(fold_scores):
                print(f'Seed {init_seed + fold_num} ==> {key} {metric} = {scores:.3f}')

    print("args:", args)
    return mean_score, std_score

