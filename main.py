import os
from XACs.dataset import MoleculeDataset, pack_data
from XACs.train import run_training, evaluate
from XACs.evaluate import run_evaluation
from XACs.GNN import GNN
from XACs.evaluate import evaluate_gnn_explain_direction, evaluate_rf_explain_direction, run_evaluation
from cross_validation import cross_validate
from XACs.utils.utils import set_seed, load_checkpoint, load_pickle 
from XACs.utils.parsing import get_args
from XACs.utils.const import SEARCH_SPACE
import torch
from hypertune import hyperopt_search, grid_search
from hyperopt import space_eval
import collections
import numpy as np
from copy import deepcopy
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == '__main__':
    args = get_args()
    sim_thre = args.sim_threshold
    dataset = MoleculeDataset(args.dataset, args.data_dir)   
    dataset.get_cliffs(args.sim_struct if args.sim_struct== 'mmp' else (args.sim_struct, args.sim_threshold), args.dist_threshold)
    args.num_node_features=dataset.num_node_features
    args.num_edge_features=dataset.num_edge_features 
    args.minimize_score = args.metric in ['rmse', 'mae']
    if args.use_gnn_opt_params: 
        config_file = os.path.join(args.config_dir, f"{args.dataset}.pkl")
        SEARCH_SPACE = SEARCH_SPACE[args.conv_name]
        if os.path.exists(config_file):
            best_params = load_pickle(config_file)
            for arg in SEARCH_SPACE.keys():
                if arg == 'hidden_dim' and args.conv_name == 'nn':
                    setattr(args, 'node_hidden_dim', space_eval(SEARCH_SPACE, best_params)[arg])
                    setattr(args, 'edge_hidden_dim', space_eval(SEARCH_SPACE, best_params)[arg])
                    setattr(args, 'hidden_dim', space_eval(SEARCH_SPACE, best_params)[arg])
                else:
                    setattr(args, arg, space_eval(SEARCH_SPACE, best_params)[arg])
                print(f"{arg}: {getattr(args, arg)}")
            print(f"Best parameters for {args.dataset} loaded from {config_file}!")
        else:
            print(f"Best parameters for {args.dataset} not found! Using default parameters...")
    if args.use_opt_xweight:
        config_file_exweight = os.path.join(args.config_dir, f"{args.dataset}_exweight.pkl")
        if os.path.exists(config_file_exweight):
            best_params = load_pickle(config_file_exweight)
            print(f"Best explanation weight for {args.dataset} loaded from {config_file_exweight}!")
            setattr(args, 'com_loss_weight', best_params['weight'])
            setattr(args, 'uncom_loss_weight', best_params['weight'])
            print(f"com_loss_weight: {args.com_loss_weight}")
            print(f"uncom_loss_weight: {args.uncom_loss_weight}")
            
        else:
            print(f"Best explanation weight for {args.dataset} not found! Using default parameters...")
    print("args:", args)
    if args.mode == 'cross_validation':
        print(f"Running cross_validation... for {args.dataset} using {args.loss}\n")
        mean_score, std_score = cross_validate(args, dataset)
        
    if args.mode == 'train_test':
        set_seed(seed=args.seed)
        data_train, data_val, data_test = dataset.split_data(split_ratio=args.split, 
                                                            split_method=args.split_method,
                                                            seed=42, 
                                                            save_split=True)
        if args.loss != 'MSE':
            data_train = pack_data(data_train, dataset.cliff_dict)
        # mols in test set is searched within the whole dataset for cliff pairs
        data_test = pack_data(data_test, dataset.cliff_dict, space=dataset.data_all)
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
                    attribute_to_last_layer=args.attribute_to_last_layer,
                    uncom_pool=args.uncom_pool,
                    normalize_att=args.normalize_att,
                    ifp=args.ifp,
                    )
        print("Total number of trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        print(f"Running GNN training... for {args.dataset} using {args.loss}\n")
        _ = run_training(args, model, data_train, data_val)
        print("Testing...")
        best_model = load_checkpoint(args)
        test_score, test_cliff_score, _ = run_evaluation(args, best_model, data_test)


    if args.mode == 'test':
        set_seed(seed=args.seed)
        model = load_checkpoint(args)
        _, _, data_test = dataset.split_data(split_ratio=args.split, 
                                            split_method=args.split_method,
                                            seed=args.seed, 
                                            save_split=True)
        gnn_score, _ = evaluate_gnn_explain_direction(dataset, data_test, model)

        data_test = pack_data(data_test, dataset.cliff_dict, space=dataset.data_all)
        test_score, test_cliff_score, _ = run_evaluation(args, model, data_test)
        
    if args.mode == 'hypertune':
        set_seed(seed=args.seed)
        args.save_checkpoints = False
        data_train, data_val, _ = dataset.split_data(split_ratio=args.split, 
                                                    split_method=args.split_method,
                                                    seed=args.seed, 
                                                    save_split=True)
        print("Running hyperopt search...")
        if args.tune_type == 'grid_search':
            data_train = pack_data(data_train, dataset.cliff_dict)
            best_params = grid_search(args, data_train, data_val)
        elif args.tune_type == 'hyperopt_search':
            best_params = hyperopt_search(args, data_train, data_val)
    
    if args.mode == 'cross_test':
        init_seed = args.seed
        all_scores = collections.defaultdict(list)
        for fold_num in range(args.num_folds):
            print(f'Fold {fold_num}')
            current_args = deepcopy(args)
            current_args.seed = init_seed + fold_num
            set_seed(seed=current_args.seed)
            current_args.checkpoint_path = os.path.join(args.model_dir, args.dataset, f'{args.dataset}_{args.loss}_model_{current_args.seed}.pt')  
            best_model = load_checkpoint(current_args)
            data_train, data_val, data_test = dataset.split_data(split_ratio=current_args.split, 
                                                                split_method=current_args.split_method,
                                                                seed=42, 
                                                                save_split=True)
            gnn_score, _ = evaluate_gnn_explain_direction(dataset, data_test, best_model)
            # merge gnn_score with all_scores
            for key, value in gnn_score.items():
                all_scores[key].append(value)
                                
            data_test = pack_data(data_test, dataset.cliff_dict, space=dataset.data_all)                             
            test_score, test_cliff_score, explan_acc = run_evaluation(current_args, best_model, data_test)
            all_scores['gnn_test_score'].append(test_score)
            all_scores['gnn_test_cliff_score'].append(test_cliff_score)
            all_scores['gnn_explanation_accuracy'].append(explan_acc)

        print(f'{args.num_folds}-fold cross validation')
        for key, fold_scores in all_scores.items():
            metric = '_' + args.metric if key=='gnn_test_score' or key=='gnn_test_cliff_score' else ''
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            print(f'{args.dataset} ==> {key}{metric} = {mean_score:.3f} +/- {std_score:.3f}')
            if args.show_individual_scores:
                for fold_num, scores in enumerate(fold_scores):
                    print(f'Seed {init_seed + fold_num} ==> {key} {metric} = {scores:.3f}')