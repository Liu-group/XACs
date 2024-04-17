import os
import random
import inspect
from argparse import Namespace
import numpy as np
from typing import List, Dict, Tuple
import pickle
import torch
from dataset import MoleculeDataset
from cliffs import ActivityCliffs, get_tanimoto_matrix
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from GNN import GNN


def set_seed(seed):
    """Sets initial seed for random numbers."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.use_deterministic_algorithms(True)

def pairwise_ranking_loss(x, y, reduction: str = 'sum'):
    """
    Computes the pairwise ranking loss between the predicted attributions and the ground truth labels.
    """
    if reduction == 'sum':
        return torch.max(-x * y, torch.zeros_like(x)).sum()
    elif reduction == 'mean':
        return torch.max(-x * y, torch.zeros_like(x)).mean()
    else:
        return torch.max(-x * y, torch.zeros_like(x))

def get_batch_indices(batch_list):
    """
    Returns a list of the indices of a data.batch
    e.g. batch = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    return [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
    """
    indices = []
    for i in range(max(batch_list)):
        indices.append(list(range(batch_list.index(i), batch_list.index(i+1))))
    indices.append(list(range(batch_list.index(max(batch_list)), len(batch_list))))
    return indices

def save_pickle(obj, path):
    """
    Saves an object as a pickle file.
    """
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    """
    Loads a pickle file as an object.
    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj

def get_model_args(args):
    """
    Returns the arguments relevant to the GNN model.
    """
    model_args = inspect.getfullargspec(GNN.__init__).args
    model_args.remove('self')
    return model_args

def load_checkpoint(current_args: Namespace):
    """
    Loads a model checkpoint.
    """
    checkpoint_path = os.path.join(current_args.save_dir, f'{current_args.dataset}_{current_args.loss}_model.pt') if current_args.checkpoint_path is None else current_args.checkpoint_path 
    assert os.path.exists(checkpoint_path), f"Checkpoint {checkpoint_path} not found"
    print(f"Loading model from {checkpoint_path}")
    if current_args.gpu is not None:
        state = torch.load(checkpoint_path)
    else:
        state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    args, model_state_dict = state['args'], state['state_dict']
    model_ralated_args = get_model_args(current_args)
    if current_args is not None:
        for key, value in vars(args).items():
            if key in model_ralated_args:
                setattr(current_args, key, value)
    else:
        current_args = args
    # Build model
    model = GNN(num_node_features=current_args.num_node_features, 
                    num_edge_features=current_args.num_edge_features,
                    num_classes=current_args.num_classes,
                    conv_name=current_args.conv_name,
                    num_layers=current_args.num_layers,
                    hidden_dim=current_args.hidden_dim,
                    dropout_rate=current_args.dropout_rate,)
    model.load_state_dict(model_state_dict)

    return model

def save_checkpoint(path: str,
                    model,
                    args: Namespace = None):
    """
    Saves a model checkpoint.

    :param model: A MPNN.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'state_dict': model.state_dict(),
    }
    torch.save(state, path)
    print(f"Model saved to {path}")

def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)