from XACs.utils.utils import set_seed, load_pickle, save_pickle
from XACs.utils.parsing import get_args
from XACs.utils.const import SEARCH_SPACE
from XACs.train import run_training
from XACs.GNN import GNN
from cross_validation import cross_validate
import torch
from itertools import product
from functools import partial
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.early_stop import no_progress_loss
import numpy as np
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
    SEARCH_SPACE = [0.05, 0.045, 0.04, 0.035, 0.03, 0.025, 0.02, 0.015, 0.01, 0.005]#, 0.001, 0.1]
    best_score = float('inf') if args.minimize_score else -float('inf')
    for w in SEARCH_SPACE:
        args.com_loss_weight = args.uncom_loss_weight = w
        args.loss = "MSE+direction"
        print(f"Running training for {args.dataset} using {args.loss} with explanation weight {w}")
        set_seed(seed=args.seed)
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
                    ifp=args.ifp,
                )
        with HiddenPrints():
            score = run_training(args, model, data_train, data_val)
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
    space = SEARCH_SPACE[args.conv_name]
    def objective(params, args, data_train, data_val):
        for key, value in params.items():
            setattr(args, key, value)
        set_seed(seed=args.seed)
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
                    ifp=args.ifp,
            )
        print(f"Tuning {args.conv_name} using {args.loss} with node_f: {args.node_hidden_dim},"
                                            f" edge_f: {args.edge_hidden_dim},"
                                            f" p: {args.dropout_rate},"
                                            f" lr: {args.lr},"
                                            f" batch_size: {args.batch_size},"
                                            f" decay: {args.weight_decay},"
                                            f" layer: {args.num_layers},"
                                            f" hd: {args.hidden_dim},"
                                            f" pool: {args.pool},"
                                            f" heads: {args.heads}")
        #print("Total number of trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        with HiddenPrints():
            try:
                score = run_training(args, model, data_train, data_val)
                score = score if args.minimize_score else -score
            except RuntimeError:
                score = float('inf') if args.minimize_score else 0
        del model, data_train, data_val
        gc.collect()
        torch.cuda.empty_cache()
        return {"loss": score, "status": STATUS_OK}
    trials = Trials()
    objective_func = partial(objective, args=args, data_train=data_train, data_val=data_val)
    best = fmin(objective_func, space, algo=tpe.suggest, max_evals=args.max_evals, trials=trials, early_stop_fn=no_progress_loss(args.hpt_patience), rstate=np.random.default_rng(args.seed))
    print(f"Best parameters: {best}")
    # save best parameters
    config_path = os.path.join(args.config_dir, f"{args.dataset}.pkl")
    save_pickle(best, config_path)
    print(f"Best parameters saved at {config_path}!")
    return best

if __name__ == '__main__':
    # load config
    args = get_args()
    set_seed(seed=args.seed)
    best_params = load_pickle(os.path.join(args.config_dir, f"{args.dataset}.pkl"))
    # load data
    SEARCH_SPACE = MPNN_SEARCH_SPACE if args.conv_name == 'nn' else GINE_SEARCH_SPACE
    for arg in SEARCH_SPACE.keys():
        setattr(args, arg, SEARCH_SPACE[arg][best_params[arg]])
    print(args)
