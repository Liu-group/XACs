import os
import numpy as np
import collections
import torch
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict
from XACs.utils import makedirs, set_seed, load_checkpoint
from XACs.dataset import MoleculeDataset, pack_data
from XACs.GNN import GNN
from XACs.train import run_training
from XACs.evaluate import evaluate_gnn_explain_direction, evaluate_rf_explain_direction, run_evaluation
from copy import deepcopy

def cross_validate(args, dataset: MoleculeDataset):
    init_seed = args.seed
    save_dir = args.save_dir
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
        data_test = pack_data(data_test, dataset.cliff_dict, space=dataset.data_all)
        model = GNN(num_node_features=current_args.num_node_features, 
                    num_edge_features=current_args.num_edge_features,
                    num_classes=current_args.num_classes,
                    conv_name=current_args.conv_name,
                    num_layers=current_args.num_layers,
                    hidden_dim=current_args.hidden_dim,
                    dropout_rate=current_args.dropout_rate,
                    )
        # get the number of parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of trainable params: ", total_params)    
        best_val_score = run_training(current_args, model, data_train, data_val)
        best_model = load_checkpoint(current_args) if current_args.save_checkpoints else model
        test_score, test_cliff_score, explan_acc = run_evaluation(current_args, best_model, data_test)
        #gnn_score, _ = evaluate_gnn_explain_direction(dataset, data_test, model)
        all_scores['gnn_test_score'].append(test_score)
        all_scores['gnn_test_cliff_score'].append(test_cliff_score)
        all_scores['gnn_explanation_accuracy'].append(explan_acc)
        #all_scores['gnn_gradcam_direction_score'].append(gnn_score['gradcam'])
        #all_scores['gnn_inputxgrad_direction_score'].append(gnn_score['inputxgrad'])
        #all_scores['gnn_graph_mask_direction_score'].append(gnn_score['mask'])

        ## not sure if necessary; the reset_parameters() function should also be checked.
        del best_model, model
        torch.cuda.empty_cache()
    # Report scores for each fold
    print(f'{args.num_folds}-fold cross validation')

    for key, fold_scores in all_scores.items():
        metric = args.metric if key!='gnn_direction_score' and key!='rf_direction_score' else ''
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f'{args.dataset} ==> {key} {metric} = {mean_score:.6f} +/- {std_score:.6f}')
        if args.show_individual_scores:
            for fold_num, scores in enumerate(fold_scores):
                print(f'Seed {init_seed + fold_num} ==> {key} {metric} = {scores:.6f}')

    print("args:", args)
    return mean_score, std_score

