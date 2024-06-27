from sklearn.ensemble import RandomForestRegressor
from rf_utils import featurize_ecfp4
import numpy as np
import os
import multiprocessing
import collections
from copy import deepcopy
from parsing import get_args
from argparse import Namespace
from utils import set_seed
from dataset import MoleculeDataset
from rdkit.Chem import MolFromSmiles
from metrics import get_metric_func
from evaluate import evaluate_rf_explain_direction


N_TREES_LIST = [100, 250, 500, 1000]
N_JOBS = int(os.getenv("LSB_DJOB_NUMPROC", multiprocessing.cpu_count()))
print("Number of Jobs: ", N_JOBS)

def train_test_rf(args: Namespace, data):
    data_train, data_val, data_test = data
    smiles_train = [data_train[i].smiles for i in range(len(data_train))]
    smiles_val = [data_val[i].smiles for i in range(len(data_val))]
    smiles_test = [data_test[i].smiles for i in range(len(data_test))]

    y_train = [data_train[i].target for i in range(len(data_train))]
    y_val = [data_val[i].target for i in range(len(data_val))]
    y_test = [data_test[i].target for i in range(len(data_test))]

    cliff_mols_test = [data_test[i].cliff for i in range(len(data_test))]
    
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

if __name__ == '__main__':
    args = get_args()
    init_seed = args.seed
    args.save_dir = f'./{args.data_path}/{args.dataset}'
    dataset = MoleculeDataset(args.dataset, args.data_path)   
    # Run training with different random seeds for each fold
    all_scores = collections.defaultdict(list)
    for fold_num in range(args.num_folds):
        print(f'Fold {fold_num}')
        current_args = deepcopy(args)
        current_args.seed = init_seed + fold_num
        set_seed(seed=current_args.seed)
        data = dataset.cliff_split(split_ratio=current_args.split, seed=current_args.seed, save_split=True)
        rf_model, rf_test_score, rf_test_cliff_score = train_test_rf(current_args, data)
        rf_score = evaluate_rf_explain_direction(dataset.cliff_dict, data[2], rf_model)
        all_scores['rf_test_score'].append(rf_test_score)
        all_scores['rf_test_cliff_score'].append(rf_test_cliff_score)
        all_scores['rf_direction_score'].append(rf_score)
        
    for key, fold_scores in all_scores.items():
        metric = args.metric if key!='gnn_direction_score' and key!='rf_direction_score' else ''
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        print(f'{args.dataset} ==> {key} {metric} = {mean_score:.6f} +/- {std_score:.6f}')
        if args.show_individual_scores:
            for fold_num, scores in enumerate(fold_scores):
                print(f'Seed {init_seed + fold_num} ==> {key} {metric} = {scores:.6f}')

    print("args:", args)
