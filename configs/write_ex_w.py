import sys
sys.path.append('../')
from utils import save_pickle, load_pickle
import os



if __name__ == '__main__':
    best_params = 0.01
    dataset = 'CHEMBL214_Ki'
    config_file_exweight = os.path.join('./', f"{dataset}_exweight.pkl")
    old_best_params = load_pickle(config_file_exweight)
    print(old_best_params)
    print(f"Old best explanation weight: {old_best_params['weight']}")
    save_pickle({'weight': best_params}, os.path.join('./', f"{dataset}_exweight.pkl"))
    print("Best parameters saved!")
    print("Checking")
    best_params = load_pickle(config_file_exweight)
    print(best_params['weight'])
