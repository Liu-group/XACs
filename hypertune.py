from parsing import get_args
from dataset import MoleculeDataset
from train import run_training
from GNN import GNN
from cross_validation import cross_validate
from utils import set_seed
import torch
from itertools import product
from functools import partial
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss
import numpy as np
from utils import save_pickle, load_pickle
torch.set_default_tensor_type(torch.DoubleTensor)
import os, sys
import gc
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# grid search
def grid_search(args, data_train, data_val):
    SEARCH_SPACE = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    best_score = float('inf') if args.minimize_score else -float('inf')
    for w in SEARCH_SPACE:
        args.com_loss_weight = args.uncom_loss_weight = w
        args.loss = "MSE+direction"
        print(f"Running training for {args.dataset} using {args.loss} with explanation weight {w}")
        set_seed(seed=args.seed)
        model = GNN(num_node_features=args.num_node_features, 
                num_edge_features=args.num_edge_features,
                num_classes=args.num_classes,
                conv_name=args.conv_name,
                num_layers=args.num_layers,
                hidden_dim=args.hidden_dim,
                dropout_rate=args.dropout_rate)
        #with HiddenPrints():
        score = run_training(args, model, data_train, data_val)
        score = score if args.minimize_score else -score
        print(f"Score: {score}")
        del model
        #gc.collect()
        #torch.cuda.empty_cache()
        if args.minimize_score and score < best_score or \
                not args.minimize_score and score > best_score:
            best_score = score
            best_params = w
    print(f"Best parameters: {best_params}")
    # save best parameters
    save_pickle({"weight": best_params}, os.path.join(args.config_dir, f"{args.dataset}_exweight.pkl"))
    print("Best parameters saved!")
    return best_params

# hyperparameters tuning using hyperopt
def hyperopt_search(args, data_train, data_val):
    space = {
        "dropout_rate": hp.choice("dropout_rate", [0., 0.2, 0.5]),
        "lr": hp.choice("lr", [1e-3, 3e-4, 1e-4]),
        "weight_decay": hp.choice("weight_decay", [0., 1e-3, 1e-4]),
        "num_layers": hp.choice("num_layers", [2, 3, 5]),
        "hidden_dim": hp.choice("hidden_dim", [64, 128]),
        "batch_size": hp.choice("batch_size", [32, 64]), 
        "pool": hp.choice("pool", ["mean", "add"]),
    }
    def objective(params, args, data_train, data_val):
        args.lr = params["lr"]
        args.weight_decay = params["weight_decay"]
        args.num_layers = params["num_layers"]
        args.hidden_dim = params["hidden_dim"]
        args.batch_size = params["batch_size"]  
        args.pool = params["pool"]
        args.dropout_rate = params["dropout_rate"]
        set_seed(seed=args.seed)
        model = GNN(num_node_features=args.num_node_features, 
            num_edge_features=args.num_edge_features,
            num_layers=params["num_layers"],
            hidden_dim=params["hidden_dim"],
            dropout_rate=params["dropout_rate"],
            pool=params["pool"],
            )
        print(f"Running training for {args.dataset} using {args.loss} with dp: {args.dropout_rate} and lr: {args.lr} and batch_size: {args.batch_size} and decay: {args.weight_decay} and layer:{args.num_layers} and hd: {args.hidden_dim} and pool: {args.pool}")
        with HiddenPrints():
            score = run_training(args, model, data_train, data_val)
        score = score if args.minimize_score else -score
        del model, data_train, data_val
        gc.collect()
        torch.cuda.empty_cache()
        return {"loss": score, "status": STATUS_OK}
    trials = Trials()
    objective_func = partial(objective, args=args, data_train=data_train, data_val=data_val)
    best = fmin(objective_func, space, algo=tpe.suggest, max_evals=args.max_evals, trials=trials, early_stop_fn=no_progress_loss(args.hpt_patience), rstate=np.random.default_rng(args.seed))
    print(f"Best parameters: {best}")
    # save best parameters
    save_pickle(best, os.path.join(args.config_dir, f"{args.dataset}.pkl"))
    print("Best parameters saved!")
    return best

if __name__ == '__main__':
    # load config
    args = get_args()
    set_seed(seed=args.seed)
    best_params = load_pickle(os.path.join(args.config_dir, f"{args.dataset}.pkl"))
    # load data
    for arg in SEARCH_SPACE.keys():
        setattr(args, arg, SEARCH_SPACE[arg][best_params[arg]])
    print(args)
