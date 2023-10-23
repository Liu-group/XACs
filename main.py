from parsing import get_args
from dataset import MoleculeDataset
from train import run_training, train_test_rf
from GNN import GNN
from evaluate import evaluate_gnn_explain_direction, evaluate_rf_explain_direction
from cross_validation import cross_validate
from utils import set_seed
import torch
torch.set_default_tensor_type(torch.DoubleTensor)


if __name__ == '__main__':
    args = get_args()
    if 'direction' in args.loss:
        args.show_direction_loss = True
    args.save_dir = f'./QSAR_ACs/{args.dataset}'
    data = MoleculeDataset(args.dataset) 
    if args.mode == 'cross_validation':
        print(f"Running cross_validation... for {args.dataset} using {args.loss}\n")
        mean_score, std_score = cross_validate(args, data)
        

    if args.mode == 'train_test':
        set_seed(seed=args.seed)
        print(f"Processsing Dataset: {args.dataset}")
        data(seed=args.seed, threshold=args.threshold, save_split = True, concat=args.show_direction_loss)

        model = GNN(num_node_features=data.num_node_features, 
                    num_edge_features=data.num_edge_features,
                    num_classes=args.num_classes,
                    conv_name=args.conv_name,
                    num_layers=args.num_layers,
                    hidden_dim=args.hidden_dim,
                    dropout_rate=args.dropout_rate,)
        print("Total number of trainable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

        print(f"Running GNN training... for {args.dataset} using {args.loss}\n")
        model, test_score, test_cliff_score = run_training(args, model, data)
        
        if args.contrast2rf:
            print("Running Random Forest...")
            model_rf, rf_test_score, rf_test_cliff_score = train_test_rf(args, data)
            

            rf_score = evaluate_rf_explain_direction(data, model_rf)
        print("Testing explainability...")
        gnn_score = evaluate_gnn_explain_direction(data, model)
    