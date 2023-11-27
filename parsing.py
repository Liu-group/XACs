import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--gpu", type=int, default=None, help="cuda")
    parser.add_argument('--save_dir', type=str, help='Path where results will be saved')
    parser.add_argument('--save_checkpoints', type=bool, default=False)
    parser.add_argument('--checkpoint_path', type=str, help='model path')
    parser.add_argument('--data_path', type=str, default='QSAR_ACs')
    parser.add_argument('--dataset', type=str, default='CHEMBL2835_Ki')
    parser.add_argument('--threshold', type=float, default=0.90, help='threshold for similarity')
    parser.add_argument('--mode', type=str, default='cross_validation', help='cross_validation or train_test')
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--split', type=list, default=[0.8, 0., 0.2])
    parser.add_argument('--metric', type=str, default='rmse')
    parser.add_argument('--minimize_score', type=bool, default=True)

   # cross validation
    parser.add_argument('--num_folds', type=int, default=10)
    parser.add_argument('--save_fold', type=bool, default=False)
    parser.add_argument('--show_individual_scores', type=bool, default=True)
    parser.add_argument('--contrast2rf', type=bool, default=False)
    # GNN
    parser.add_argument('--conv_name', type=str, default='nn')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--pool', type=str, default='mean')    
    # training
    parser.add_argument('--dropout_rate', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stop_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.)
    parser.add_argument('--factor', type=float, default=0.9)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--min_lr', type=int, default=1e-7)

    # explanation
    parser.add_argument('--opt_goal', type=str, default='MSE')
    parser.add_argument(
        "--loss", type=str, default="MSE", help="Type of loss for training GNN."
    ) # ['MSE', 'MSE+att', 'MSE+att+sparsity', 'MSE+sparsity', 'MSE+direction', 'MSE+direction+sparsity']
    parser.add_argument('--att_loss_weight', type=float, default=0)
    parser.add_argument('--sparsity_loss_weight', type=float, default=0.)
    parser.add_argument('--norm', type=int, default=1)
    parser.add_argument('--direction_loss_weight', type=float, default=0.01)
    parser.add_argument('--show_direction_loss', type=bool, default=False)
    parser.add_argument('--att_method', type=str, default='GradCAM')
    parser.add_argument('--return_gradients', type=bool, default=False)

    return parser.parse_args()


