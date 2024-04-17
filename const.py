from rdkit import Chem
from hyperopt import hp

TIMEOUT_MCS = 1800

ATOM_TYPES = [
        "B", 
        "C",
        "N",
        "O",
        "S",
        "F",
        "P",
        "Cl",
        "Br",
        "I",
        "*",
    ]
BOND_TYPES = ['SINGLE','DOUBLE','TRIPLE','AROMATIC']

STEREO_TYPES = ["STEREOZ", "STEREOE", "STEREONONE"]

DATASETS = ['CHEMBL4616_EC50',  'CHEMBL4792_Ki', 'CHEMBL1871_Ki', 'CHEMBL2971_Ki',
 'CHEMBL239_EC50', 'CHEMBL233_Ki','CHEMBL235_EC50', 'CHEMBL231_Ki','CHEMBL218_EC50','CHEMBL244_Ki',
 'CHEMBL234_Ki','CHEMBL237_Ki','CHEMBL1862_Ki','CHEMBL4203_Ki','CHEMBL2047_EC50',
 'CHEMBL219_Ki','CHEMBL236_Ki','CHEMBL228_Ki','CHEMBL2147_Ki','CHEMBL204_Ki',
 'CHEMBL262_Ki','CHEMBL287_Ki','CHEMBL2034_Ki','CHEMBL3979_EC50','CHEMBL238_Ki','CHEMBL2835_Ki',
 'CHEMBL4005_Ki','CHEMBL237_EC50','CHEMBL264_Ki','CHEMBL214_Ki']

SEARCH_SPACE = {
        "dropout_rate": hp.choice("dropout_rate", [0., 0.2, 0.5]),
        "lr": hp.choice("lr", [1e-3, 3e-4, 1e-4]),
        "weight_decay": hp.choice("weight_decay", [0., 1e-3, 1e-4]),
        "num_layers": hp.choice("num_layers", [2, 3, 5]),
        "batch_size": hp.choice("batch_size", [32, 64, 128]), 
        "hidden_dim": hp.choice("hidden_dim", [64, 128]),
        "pool": hp.choice("pool", ["mean", "add"]),
    }