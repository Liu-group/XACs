# add ../ to the path
import sys
sys.path.append('../')
from utils import save_pickle, load_pickle
SEARCH_SPACE = {
        "batch_size": [32, 64, 128],
        "dropout_rate": [0., 0.2, 0.5],
        "hidden_dim": [64, 128, 256],
        "lr": [1e-3, 3e-4, 1e-4],
        "num_layers": [2, 3, 5],
        "pool": ["mean", "add"],
        "weight_decay": [0., 1e-3, 1e-4],
}

if __name__ == "__main__":
    data = {'batch_size': 64, 
            'dropout_rate': 0.2, 
            'hidden_dim': 64, 
            'lr': 0.001, 
            'num_layers': 2, 
            'pool': 'mean', 
            'weight_decay': 0.}
    config_dict = {}
    dataset = 'CHEMBL3979_EC50'
    for key in SEARCH_SPACE.keys():
        config_dict[key] = SEARCH_SPACE[key].index(data[key])
    # Save the data
    save_pickle(config_dict, dataset+'.pkl')
    # Print the data
    print(config_dict)
    print(f"{dataset}.pkl saved!")