import os
import numpy as np
import collections
import torch
from sklearn.model_selection import StratifiedKFold
from typing import List, Dict
from utils import makedirs, set_seed
from dataset import MoleculeDataset
from GNN import GNN
from train import run_training, train_test_rf
from evaluate import evaluate_gnn_explain_direction, evaluate_rf_explain_direction

def cross_validate(args, data: MoleculeDataset):
    init_seed = args.seed
    save_dir = args.save_dir
    threshold = args.threshold
    # Run training with different random seeds for each fold
    all_scores = collections.defaultdict(list)
    for fold_num in range(args.num_folds):
        print(f'Fold {fold_num}')
        args.seed = init_seed + fold_num
        set_seed(seed=args.seed)
        if args.save_fold:
            args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
            makedirs(args.save_dir, exist_ok=True)
        data(seed=args.seed, threshold=threshold, save_split = True, concat=args.show_direction_loss)
        model = GNN(num_node_features=data.num_node_features, 
                    num_edge_features=data.num_edge_features,
                    num_classes=args.num_classes,
                    conv_name=args.conv_name,
                    num_layers=args.num_layers,
                    hidden_dim=args.hidden_dim,
                    dropout_rate=args.dropout_rate,
                    )
        # get the number of parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of trainable params: ", total_params)    
            
        model, test_score, test_cliff_score = run_training(args, model, data)
        gnn_score = evaluate_gnn_explain_direction(data, model)
        all_scores['gnn_test_score'].append(test_score)
        all_scores['gnn_test_cliff_score'].append(test_cliff_score)
        all_scores['gnn_direction_score'].append(gnn_score)
        if args.contrast2rf:
            rf_model, rf_test_score, rf_test_cliff_score = train_test_rf(args, data)
            rf_score = evaluate_rf_explain_direction(data, rf_model)
            all_scores['rf_test_score'].append(rf_test_score)
            all_scores['rf_test_cliff_score'].append(rf_test_cliff_score)
            all_scores['rf_direction_score'].append(rf_score)

        
        ####### testing mlp #######
        from torch_geometric.loader import DataLoader
        from sklearn.model_selection import train_test_split
        data_train_init, data_test, _ = data.data_train, data.data_test, data.batched_data_train
        data_train, _ = train_test_split(data_train_init, test_size=args.split[1]/(1.0-args.split[2]), shuffle=True, random_state=args.seed)
        train_loader = DataLoader(data_train, batch_size = 1, shuffle=False)
        test_loader = DataLoader(data_test, batch_size = 1, shuffle=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.eval().to(device)
        ft_train, y_train = [], []
        with torch.no_grad():
            for dat in train_loader:
                x, edge_index = dat.x.to(device), dat.edge_index.type(torch.LongTensor).to(device)
                edge_attr = dat.edge_attr.to(device)
                batch = dat.batch.to(device)
                y = dat.target.to(device)
                out = model(x=x, edge_attr=edge_attr, edge_index=edge_index, batch=batch, return_features=True)
                ft_train.append(out.cpu().numpy())
                y_train += y.cpu().numpy().tolist()
        ft_train = np.concatenate(ft_train, axis=0)
        ft_test, y_test = [], []
        with torch.no_grad():
            for data_test in test_loader:
                x, edge_index = data_test.x.to(device), data_test.edge_index.type(torch.LongTensor).to(device)
                edge_attr = data_test.edge_attr.to(device)
                batch = data_test.batch.to(device)
                y = data_test.target.to(device)
                out = model(x=x, edge_attr=edge_attr, edge_index=edge_index, batch=batch, return_features=True)
                ft_test.append(out.cpu().numpy())
                y_test += y.cpu().numpy().tolist()
        ft_test = np.concatenate(ft_test, axis=0)
        
        from sklearn.ensemble import RandomForestRegressor
        model_rf = RandomForestRegressor(n_estimators=100, random_state=args.seed)
        model_rf.fit(ft_train, y_train)
        y_pred = model_rf.predict(ft_test)
        from metrics import get_metric_func
        metric_func = get_metric_func(metric=args.metric)
        gnn_rf_rmse = metric_func(y_test, y_pred)
        print("Using GNN features for RF rmse:", gnn_rf_rmse)
        all_scores['gnn_rf_test_rmse'].append(gnn_rf_rmse)
    ##############################
        ## not sure if necessary; the reset_parameters() function should also be checked.
        del model
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

def k_fold_cross_validate(args, data: MoleculeDataset):
    """
    Performs k-fold cross validation on the training dataset for a given model.
    :param args: Arguments.
    :param data: MoleculeDataset.
    :return: Average test score across all folds.
    """
    save_dir = args.save_dir
    splits = split_data_k_fold(data, args.num_folds, args.seed)
    data_train = data.data_train
    all_scores = []
    for fold_num, split in enumerate(splits):
        #hyperparameters = {k: v.item() if isinstance(v, np.generic) else v for k, v in hyperparameters.items()}
        model = GNN(args=args)
        if args.save_split:
            args.save_dir = os.path.join(save_dir, f'fold_{fold_num}')
            makedirs(args.save_dir)
        tr_fold = [data_train[i] for i in split['train_idx']] if type(data_train) is list else data_train[split['train_idx']]
        val_fold = [data_train[i] for i in split['val_idx']] if type(data_train) is list else data_train[split['val_idx']]
        model_scores = run_training(model, tr_fold, val_fold, args)

        all_scores.append(model_scores)

        torch.cuda.empty_cache()

    return sum(all_scores)/len(all_scores)

def split_data_k_fold(data: MoleculeDataset,
                      n_folds: int = 5,
                      seed: int = 42) -> List[Dict[str, np.ndarray]]:
    """
    Splits data into k-fold train/test sets for cross-validation.
    """

    ss = StratifiedKFold(n_splits=n_folds, random_state=seed, shuffle=True)
    cutoff = np.median(data.y_train)
    labels = [0 if i < cutoff else 1 for i in data.y_train]
    splits = [{'train_idx': i, 'val_idx': j} for i, j in ss.split(labels, labels)]
    return splits

if __name__ == '__main__':
    from parsing import get_args
    from main import set_seed
    args = get_args()
    args.save_dir = '/home/xuchen/ACs/QSAR_ACs/CHEMBL4203_Ki'
    set_seed(seed=args.seed)
    data = MoleculeDataset(args.dataset)
    data()
    mean_score = cross_validate(args, data)
    print("mean score", mean_score)