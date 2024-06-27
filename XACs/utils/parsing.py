import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--gpu", type=int, default=None, help="cuda")
    parser.add_argument('--save_dir', type=str, help='Path where results will be saved')
    parser.add_argument('--save_checkpoints', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, help='model path')
    parser.add_argument('--config_dir', type=str, default='./configs/gine_configs') # mpnn_configs
    parser.add_argument('--data_path', type=str, default='Molecular_datasets')#'Benchmark_Data'/ 'QSAR_ACs'
    
    parser.add_argument('--dataset', type=str, default='BACE')#CHEMBL233_Ki
    parser.add_argument('--task', type=str, default='classification', help='classification or regression') 
    parser.add_argument('--num_classes', type=int, default=1) # currently only binary classification or regression
    parser.add_argument('--metric', type=str, default='auroc') 
    parser.add_argument('--minimize_score', type=bool, default=False)

    parser.add_argument('--sim_threshold', type=float, default=0.9, help='threshold for similarity')
    parser.add_argument('--dist_threshold', type=float, default=1.0, help='threshold for distance')
    parser.add_argument('--mode', type=str, default='hypertune', help='cross_validation or train_test or test or hypertune')
    parser.add_argument('--split', type=list, default=[0.8, 0.1, 0.1])
    parser.add_argument('--split_method', type=str, default='random', help='random or cliff split')

    # hypertune
    parser.add_argument('--max_evals', type=int, default=20)
    parser.add_argument('--hpt_patience', type=int, default=10)
    parser.add_argument('--use_gnn_opt_params', type=bool, default=False) # use optimal parameters
    parser.add_argument('--use_opt_xweight', type=bool, default=False) # use optimal explanation weight
    parser.add_argument('--tune_type', type=str, default='hyperopt_search') # grid_search or hyperopt_search; 
    #if grid_search, use_opt_params should be True

   # cross validation
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--save_fold', type=bool, default=False)
    parser.add_argument('--show_individual_scores', type=bool, default=True)
    # GNN
    parser.add_argument('--conv_name', type=str, default='gine')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
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
    parser.add_argument("--loss", type=str, default="MSE", help="Type of loss for training GNN.") 
    parser.add_argument('--com_loss_weight', type=float, default=0.)
    parser.add_argument('--uncom_loss_weight', type=float, default=0.)
    parser.add_argument('--normalize', type=bool, default=False)


    return parser.parse_args()


