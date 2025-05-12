import os
from XACs.dataset import MoleculeDataset, pack_data
from XACs.train import run_training, run_cv_train
from XACs.evaluate import run_evaluation, run_evaluation_ensemble
from XACs.models.GNN import GNN
from XACs.utils.utils import set_seed, load_checkpoint, load_pickle, get_deg
from XACs.utils.parsing import get_args
from XACs.utils.const import SEARCH_SPACE
from hypertune import hyperopt_search, grid_search
from hyperopt import space_eval
import collections
import numpy as np
from copy import deepcopy
import json

if __name__ == '__main__':
    args = get_args()
    sim_thre = args.sim_threshold
    dataset = MoleculeDataset(args.dataset, args.data_dir)   
    dataset.get_cliffs(args.sim_struct if args.sim_struct== 'mmp' else (args.sim_struct, args.sim_threshold), args.dist_threshold)
    args.num_node_features=dataset.num_node_features
    args.num_edge_features=dataset.num_edge_features 
    args.minimize_score = args.metric[0].lower().startswith('rmse') or args.metric[0].lower().startswith('mae')
    init_seed = args.seed
    all_scores = collections.defaultdict(list)
    for fold_num in range(1, args.num_folds+1):
        print(f'Fold {fold_num}')
        current_args = deepcopy(args)
        current_args.seed = init_seed + fold_num - 1
        data_train, data_val, data_test = dataset.split_data(split_ratio=args.split, 
                                                    split_method=args.split_method,
                                                    seed=current_args.seed, 
                                                    save_split=True)
        if args.use_gnn_opt_params: 
            config_file = os.path.join(args.config_dir, f"{args.dataset}_{current_args.seed}.pkl")
            param_space = SEARCH_SPACE[args.conv_name]
            if os.path.exists(config_file):
                best_params = load_pickle(config_file)
                print(f"Best parameters for {args.dataset} loaded from {config_file}!")
            else:
                print(f"Best parameters for {args.dataset} not found! Start hyperopt search...")
                current_args.save_checkpoints = False
                best_params = hyperopt_search(current_args, data_train, data_val)
                print(f"Best parameters for {args.dataset} loaded from hyperopt search!")
            for arg in param_space.keys():
                setattr(current_args, arg, space_eval(param_space, best_params)[arg])
                print(f"{arg}: {getattr(current_args, arg)}")
                
        if args.loss != 'MSE':
            data_train = pack_data(data_train, dataset.cliff_dict)
            if args.use_opt_xweight:
                config_file_exweight = os.path.join(args.config_dir, f"{args.dataset}_exweight.pkl")
                if os.path.exists(config_file_exweight):
                    best_params = load_pickle(config_file_exweight)
                    print(f"Best explanation weight for {args.dataset} loaded from {config_file_exweight}!")
                else:
                    print(f"Best explanation weight for {args.dataset} not found! Start hyperopt search...")
                    current_args.save_checkpoints = False
                    best_params = grid_search(current_args, data_train, data_val)
                    print(f"Best explanation weight for {args.dataset} loaded from grid search!")
                setattr(current_args, 'com_loss_weight', best_params['weight'])
                setattr(current_args, 'uncom_loss_weight', best_params['weight'])
                print(f"com_loss_weight: {current_args.com_loss_weight}")
                print(f"uncom_loss_weight: {current_args.uncom_loss_weight}")

        print("current_args:", current_args)
        current_args.save_checkpoints = True
        if args.ensemble:
            models = []
            for i, metric in enumerate(args.metric):
                for j in range(5):
                    checkpoint_path = os.path.join(args.model_dir, args.dataset, 
                                                        f'{args.dataset}_{args.loss}_model_{(init_seed + fold_num) * 10 + j}_{metric}.pt')
                    if args.save_checkpoints and os.path.exists(checkpoint_path):
                        best_model = load_checkpoint(current_args, checkpoint_path)
                    else:
                        print(f"Warning: Checkpoint not found, start training...")
                        run_cv_train(current_args, data_train)
                        best_model = load_checkpoint(current_args, checkpoint_path)
                    models.append(best_model)
                fold_scores = run_evaluation_ensemble(current_args, dataset, data_test, models, metric, xeval=True if i == 0 else False)
                for key, value in fold_scores.items():
                    all_scores[key].append(value)
        else:
            for i, metric in enumerate(args.metric):
                check_point_path = os.path.join(args.model_dir, args.dataset, 
                                                f'{args.dataset}_{args.loss}_model_{current_args.seed}_{metric}.pt')
                if os.path.exists(check_point_path):
                    model = load_checkpoint(current_args, check_point_path)
                else:
                    print(f"Warning: Checkpoint not found, start training...")
                    run_training(current_args, data_train, data_val)
                    model = load_checkpoint(current_args, check_point_path)
                fold_scores = run_evaluation(current_args, dataset, data_test, model, metric, xeval=True if i == 0 else False)
                for key, value in fold_scores.items():
                    all_scores[key].append(value)
    # Report scores for each fold
    print(f'\n{args.num_folds}-fold cross validation results:')
    for key, fold_scores in all_scores.items():
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f'{args.dataset} ==> {key} = {mean_score:.3f} +/- {std_score:.3f}')
        if args.show_individual_scores:
            for fold_num, scores in enumerate(fold_scores):
                print(f'Seed {init_seed + fold_num} ==> {key} = {scores:.3f}')    
    # save all_scores as a json file
    for key in all_scores:
        all_scores[key] = [float(val) if hasattr(val, 'item') else val for val in all_scores[key]]
    with open(os.path.join(args.model_dir, args.dataset, f'{args.dataset}_{args.num_folds}_fold_scores.json'), 'w') as f:
        json.dump(all_scores, f)