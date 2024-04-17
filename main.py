import os
from parsing import get_args
from dataset import MoleculeDataset, pack_data
from train import run_training, train_test_rf, evaluate
from evaluate import run_evaluation
from GNN import GNN
from evaluate import evaluate_gnn_explain_direction, evaluate_rf_explain_direction, run_evaluation
from cross_validation import cross_validate
from utils import set_seed, load_checkpoint
import torch
from hypertune import hyperopt_search, grid_search
from hyperopt import space_eval
from utils import load_pickle
from const import SEARCH_SPACE
import collections
import numpy as np
from copy import deepcopy

torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == '__main__':
    args = get_args()
    args.save_dir = f'./{args.data_path}/{args.dataset}'
    dataset = MoleculeDataset(args.dataset, args.data_path)   
    args.num_node_features=dataset.num_node_features
    args.num_edge_features=dataset.num_edge_features 

    if args.use_opt_params: 
        config_file = os.path.join(args.config_dir, f"{args.dataset}.pkl")
        assert os.path.exists(config_file), f"Optimal parameters for {args.dataset} not found!"
        if os.path.exists(config_file):
            best_params = load_pickle(config_file)
            #print(best_params)
            for arg in SEARCH_SPACE.keys():
                setattr(args, arg, space_eval(SEARCH_SPACE, best_params)[arg])
                print(f"{arg}: {getattr(args, arg)}")
            print(f"Best parameters for {args.dataset} loaded!")
        else:
            print(f"Best parameters for {args.dataset} not found! Using default parameters...")
        config_file_exweight = os.path.join(args.config_dir, f"{args.dataset}_exweight.pkl")
        if os.path.exists(config_file_exweight):
            best_params = load_pickle(config_file_exweight)
            print(f"Best explanation weight for {args.dataset} loaded!")
            setattr(args, 'com_loss_weight', best_params['weight'])
            setattr(args, 'uncom_loss_weight', best_params['weight'])
            print(f"com_loss_weight: {args.com_loss_weight}")
            print(f"uncom_loss_weight: {args.uncom_loss_weight}")
            
        else:
            print(f"Best explanation weight for {args.dataset} not found! Using default parameters...")

    if args.mode == 'cross_validation':
        print(f"Running cross_validation... for {args.dataset} using {args.loss}\n")
        mean_score, std_score = cross_validate(args, dataset)
        
    if args.mode == 'train_test':
        set_seed(seed=args.seed)
        print(f"Processsing Dataset: {args.dataset}")
        data_train, data_val, data_test = dataset.cliff_split(split_ratio=args.split, seed=42, save_split=True)
        if args.loss != 'MSE':
            data_train = pack_data(data_train, dataset.cliff_dict)
        # mols in test set is searched within the whole dataset for cliff pairs
        data_test = pack_data(data_test, dataset.cliff_dict, space=dataset.data_all)
        model = GNN(num_node_features=args.num_node_features, 
                    num_edge_features=args.num_edge_features,
                    num_classes=args.num_classes,
                    conv_name=args.conv_name,
                    num_layers=args.num_layers,
                    hidden_dim=args.hidden_dim,
                    dropout_rate=args.dropout_rate,)
        print("Total number of trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        print(f"Running GNN training... for {args.dataset} using {args.loss}\n")
        _ = run_training(args, model, data_train, data_val)
        print("Testing...")
        best_model = load_checkpoint(args)
        test_score, test_cliff_score, _ = run_evaluation(args, best_model, data_test)
        
        if args.contrast2rf:
            print("Running Random Forest...")
            model_rf, rf_test_score, rf_test_cliff_score = train_test_rf(args, dataset)
            
            rf_score = evaluate_rf_explain_direction(dataset, model_rf)

    if args.mode == 'test':
        set_seed(seed=args.seed)
        model = load_checkpoint(args)
        print(f"Processsing Dataset: {args.dataset}")
        _, _, data_test = dataset.cliff_split(split_ratio=args.split, seed=args.seed, save_split=True)

        #gnn_score, _ = evaluate_gnn_explain_direction(dataset, data_test, model)

        data_test = pack_data(data_test, dataset.cliff_dict, space=dataset.data_all)
        test_score, test_cliff_score, _ = run_evaluation(args, model, data_test)
        if args.contrast2rf:
            print("Running Random Forest...")
            model_rf, rf_test_score, rf_test_cliff_score = train_test_rf(args, data)
            rf_score = evaluate_rf_explain_direction(data, model_rf)
        
    if args.mode == 'hypertune':
        set_seed(seed=args.seed)
        print(f"Processsing Dataset: {args.dataset}")
        data_train, data_val, _ = dataset.cliff_split(split_ratio=args.split, seed=args.seed, save_split=True)
        print("Running hyperopt search...")
        if args.tune_type == 'grid_search':
            data_train = pack_data(data_train, dataset.cliff_dict)
            best_params = grid_search(args, data_train, data_val)
        elif args.tune_type == 'hyperopt_search':
            best_params = hyperopt_search(args, data_train, data_val)
    
    if args.mode == 'cross_test':
        init_seed = args.seed
        save_dir = args.save_dir
        threshold = args.threshold
        # Run training with different random seeds for each fold
        all_scores = collections.defaultdict(list)
        for fold_num in range(args.num_folds):
            print(f'Fold {fold_num}')
            current_args = deepcopy(args)
            current_args.seed = init_seed + fold_num
            set_seed(seed=current_args.seed)
            data_train, data_val, data_test = dataset.cliff_split(split_ratio=current_args.split, seed=42, save_split=True)
            if current_args.loss != 'MSE':
                data_train = pack_data(data_train, dataset.cliff_dict)
            data_test = pack_data(data_test, dataset.cliff_dict, space=dataset.data_all)
            current_args.checkpoint_path = os.path.join(args.save_dir, f'{args.dataset}_{args.loss}_model_{current_args.seed}_test.pt')  
            best_model = load_checkpoint(current_args)
            ##########
            from torch_geometric.loader import DataLoader
            from metrics import get_metric_func
            
            val_loader = DataLoader(data_val, batch_size = args.batch_size, shuffle=False)
            metric = args.metric
            metric_func = get_metric_func(metric=metric)
            loss_func = torch.nn.MSELoss()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            best_model.to(device)
            val_score, val_loss = evaluate(current_args, best_model, val_loader, loss_func, metric_func, device)
            print(f'Validation {args.metric} = {val_score:.6f}')
            ##########
            test_score, test_cliff_score, explan_acc = run_evaluation(current_args, best_model, data_test)
            all_scores['gnn_test_score'].append(test_score)
            all_scores['gnn_test_cliff_score'].append(test_cliff_score)
            all_scores['gnn_explanation_accuracy'].append(explan_acc)
            print()
            if current_args.contrast2rf:
                rf_model, rf_test_score, rf_test_cliff_score = train_test_rf(current_args, data)
                rf_score = evaluate_rf_explain_direction(data, rf_model)
                all_scores['rf_test_score'].append(rf_test_score)
                all_scores['rf_test_cliff_score'].append(rf_test_cliff_score)
                all_scores['rf_direction_score'].append(rf_score)

            del best_model
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
