import argparse

def str_to_bool(v):
    """
    Convert string to boolean for argparse.
    Handles 'True', 'False', 'true', 'false', '1', '0', etc.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got: {v}')

def get_args():
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--gpu", type=int, default=None, help="cuda")
    parser.add_argument('--save_checkpoints', type=str_to_bool, default=True)
    parser.add_argument('--model_dir', type=str, default='Benchmark_Data') #Benchmark_Data
    parser.add_argument('--config_dir', type=str, default='./configs/nn_configs') # mpnn_configs
    parser.add_argument('--data_dir', type=str, default='Data')#'Benchmark_Data' or 'QSAR_ACs' or 'Data'
    
    parser.add_argument('--dataset', type=str, default='CHEMBL234_Ki')#CHEMBL233_Ki
    parser.add_argument('--task', type=str, default='regression', help='classificfation or regression') 
    parser.add_argument('--num_classes', type=int, default=1) # currently only binary classification or regression
    parser.add_argument('--metric', type=str, nargs='+', default=['rmse']) 

    parser.add_argument('--sim_threshold', type=float, default=0.9, help='threshold for similarity')
    parser.add_argument('--dist_threshold', type=float, default=1.0, help='fold-wise threshold for distance')
    parser.add_argument('--dict_path', type=str, default=None)
    parser.add_argument('--split', type=list, default=[0.8, 0.1, 0.1])
    parser.add_argument('--split_method', type=str, default='random', help='random or cliff split')

    # hypertune
    parser.add_argument('--max_evals', type=int, default=100)
    parser.add_argument('--hpt_patience', type=int, default=20)
    parser.add_argument('--use_gnn_opt_params', type=str_to_bool, default=True) # use optimal parameters
    parser.add_argument('--use_opt_xweight', type=str_to_bool, default=True) # use optimal explanation weight
    parser.add_argument('--tune_type', type=str, default='hyperopt_search') # hyperopt_search or grid_search; if grid_search, use_opt_params should be True

   # cross validation
    parser.add_argument('--num_folds', type=int, default=1, help='number of folds for cross validation testing')
    parser.add_argument('--show_individual_scores', type=str_to_bool, default=True)
    # GNN
    parser.add_argument('--ifp', type=str_to_bool, default=False)
    parser.add_argument('--num_node_features', type=int, default=42)
    parser.add_argument('--num_edge_features', type=int, default=6)
    parser.add_argument('--node_hidden_dim', type=int, default=128)
    parser.add_argument('--edge_hidden_dim', type=int, default=128)
    parser.add_argument('--conv_name', type=str, default='gine') # nn, gine, gat
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--pool', type=str, default='add') 
    parser.add_argument('--attribute_to_last_layer', type=str_to_bool, default=True)   
    parser.add_argument('--ensemble', type=str_to_bool, default=False, help='enable ensemble evaluation (uses multiple checkpoints)')
    parser.add_argument('--embed_method', type=str, default='linear',
                        help='embedding method node/edge features: linear, ifp, or original features')
    # training
    parser.add_argument('--dropout_rate', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--early_stop_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--factor', type=float, default=0.999)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--min_lr', type=int, default=1e-7)

    # explanation
    parser.add_argument('--sim_struct', type=str, default='combined')# combined or mmp
    parser.add_argument("--loss", type=str, default="MSE", help="Type of loss for training GNN.") 
    parser.add_argument('--com_loss_weight', type=float, default=0.001)
    parser.add_argument('--uncom_loss_weight', type=float, default=0.001)
    parser.add_argument('--uncom_pool', type=str, default='add')
    parser.add_argument('--normalize_att', type=str_to_bool, default=False)
    parser.add_argument('--gnes', type=str_to_bool, default=False)
    parser.add_argument('--att_method', type=str, default=None, help='attribution head for explanation loss; use "mlp" to learn node attributions, or None to use gradient-based explanations')
    parser.add_argument('--xscheduler', type=str_to_bool, default=False)
    parser.add_argument('--pair_cap', type=int, default=8, help='Maximum number of cliff pairs to pack per molecule (None keeps all)')
    parser.add_argument('--xeval', type=str_to_bool, default=False)
    return parser.parse_args()


