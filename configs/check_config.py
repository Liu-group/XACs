import sys
sys.path.append('../')
from const import DATASETS
from utils import load_pickle
import os
from hyperopt import space_eval
from const import SEARCH_SPACE




if __name__ == "__main__":
    config_dir = './'

    for dataset in DATASETS:
        config_file = os.path.join(config_dir, f"{dataset}.pkl")
        assert os.path.exists(config_file), f"Optimal parameters for {args.dataset} not found!"
        if os.path.exists(config_file):
            best_params = load_pickle(config_file)
            #print(best_params)
            for arg in SEARCH_SPACE.keys():
                if arg == 'hidden_dim' and space_eval(SEARCH_SPACE, best_params)[arg]==256:
                    print(dataset)
