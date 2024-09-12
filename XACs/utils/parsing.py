import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--gpu", type=int, default=None, help="cuda")
    parser.add_argument('--save_checkpoints', type=bool, default=True)
    parser.add_argument('--model_dir', type=str, default='Benchmark_Data') #Benchmark_Data
    parser.add_argument('--config_dir', type=str, default='./configs/nn_configs') # mpnn_configs
    parser.add_argument('--data_dir', type=str, default='Data')#'Benchmark_Data' or 'QSAR_ACs' or 'Data'
    
    parser.add_argument('--dataset', type=str, default='CHEMBL236_Ki')#CHEMBL233_Ki
    parser.add_argument('--task', type=str, default='regression', help='classificfation or regression') 
    parser.add_argument('--num_classes', type=int, default=1) # currently only binary classification or regression
    parser.add_argument('--metric', type=str, default='rmse') 

    parser.add_argument('--sim_threshold', type=float, default=0.9, help='threshold for similarity')
    parser.add_argument('--dist_threshold', type=float, default=1.0, help='threshold for distance')
    parser.add_argument('--mode', type=str, default='cross_test', help='cross_validation or train_test or test or hypertune or cross_test')
    parser.add_argument('--split', type=list, default=[0.8, 0.1, 0.1])
    parser.add_argument('--split_method', type=str, default='scaffold', help='random or cliff split')

    # hypertune
    parser.add_argument('--max_evals', type=int, default=50)
    parser.add_argument('--hpt_patience', type=int, default=10)
    parser.add_argument('--use_gnn_opt_params', type=bool, default=True) # use optimal parameters
    parser.add_argument('--use_opt_xweight', type=bool, default=True) # use optimal explanation weight
    parser.add_argument('--tune_type', type=str, default='hyperopt_search') # hyperopt_search or grid_search; 
    #if grid_search, use_opt_params should be True

   # cross validation
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--show_individual_scores', type=bool, default=True)
    # GNN
    parser.add_argument('--ifp', type=bool, default=True)
    parser.add_argument('--num_node_features', type=int, default=42)
    parser.add_argument('--num_edge_features', type=int, default=6)
    parser.add_argument('--node_hidden_dim', type=int, default=128)
    parser.add_argument('--edge_hidden_dim', type=int, default=128)
    parser.add_argument('--conv_name', type=str, default='nn') # nn, gine, gat
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--pool', type=str, default='add') 
    parser.add_argument('--attribute_to_last_layer', type=bool, default=True)   
    # training
    parser.add_argument('--dropout_rate', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--early_stop_epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--factor', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_lr', type=int, default=1e-7)

    # explanation
    parser.add_argument('--sim_struct', type=str, default='mmp')# combined or mmp
    parser.add_argument("--loss", type=str, default="MSE+direction", help="Type of loss for training GNN.") 
    parser.add_argument('--com_loss_weight', type=float, default=0.001)
    parser.add_argument('--uncom_loss_weight', type=float, default=0.001)
    parser.add_argument('--uncom_pool', type=str, default='add')
    parser.add_argument('--normalize_att', type=bool, default=False)
    parser.add_argument('--gnes', type=bool, default=False)
    parser.add_argument('--xscheduler', type=bool, default=False)

    return parser.parse_args()


