import os
from const import SEARCH_SPACE
from const import DATASETS
from utils import load_pickle
config_dir = './configs'

for dataset_name in DATASETS:
    config_file = os.path.join(config_dir, f"{dataset_name}.pkl")
    assert os.path.exists(config_file), f"Optimal parameters for {args.dataset} not found!"
    if os.path.exists(config_file):
        best_params = load_pickle(config_file)
        #print(best_params)
        print(f"Best parameters for {dataset_name} loaded!")
    else:
        print(f"Best parameters for {dataset_name} not found! Using default parameters...")
    config_file_exweight = os.path.join(config_dir, f"{dataset_name}_exweight.pkl")
    if os.path.exists(config_file_exweight):
        best_params = load_pickle(config_file_exweight)
        print(f"Best explanation weight for {dataset_name} loaded!")
        print(f"com_loss_weight: {best_params['weight']}")
