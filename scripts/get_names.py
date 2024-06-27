import os 
import sys
from const import DATASETS
from utils import load_pickle
config_dir = './configs'
for ds in DATASETS:
    config_file_exweight = os.path.join(config_dir, f"{ds}_exweight.pkl")
    if os.path.exists(config_file_exweight):
        best_params = load_pickle(config_file_exweight)
        if best_params['weight'] == 0.005:
            print(ds)