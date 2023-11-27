import os
import random
import torch
import inspect
from argparse import Namespace
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple

from dataset import MoleculeDataset
from cliffs import ActivityCliffs, get_tanimoto_matrix
from sklearn.cluster import SpectralClustering
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

def pairwise_ranking_loss(x, y):
    """
    Computes the pairwise ranking loss between the predicted attributions and the ground truth labels.
    """
    # Compute the pairwise ranking loss
    loss = 0.
    #loss += torch.max(-x * y, torch.zeros_like(x)).sum()
    loss += torch.max(-x * y, -torch.ones_like(x)).sum()
    #loss += (-x * y).sum()
    return loss

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
    assert os.path.exists(current_args.checkpoint_path), "Checkpoint not found"
    if current_args.gpu is not None:
        state = torch.load(current_args.checkpoint_path)
    else:
        state = torch.load(current_args.checkpoint_path, map_location=torch.device('cpu'))
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


def split_data(data: MoleculeDataset, 
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 42,
               args: Namespace = None) -> Tuple[MoleculeDataset, MoleculeDataset, MoleculeDataset]:
    """
    Splits data into train/validation/test sets.
    
    :param data: Dataset to split.
    :param split_type: Split type. Can be one of ['random', 'scaffold_balanced'].
    :param sizes: Dataset split sizes. Tuple of three floats (train_size, val_size, test_size).
    :param seed: Random seed.
    :param args: Arguments.
    :return: A tuple containing the train, validation, and test splits of the data.
    """
    assert len(sizes) == 3 and sum(sizes) == 1
    if split_type == 'cliffs_balanced':
        return split_cliffs_balanced(data, sizes, seed)





def split_cliffs_balanced(data: MoleculeDataset,
                          sizes: Tuple[float, float, float] = (0.8, 0.2, 0),
                          seed: int = 42,
                          in_log10: bool = True, 
                          n_clusters: int = 5,
                          similarity: float = 0.9, 
                          potency_fold: int = 10, 
                          remove_stereo: bool = False):
    """ Split data into train/test according to activity cliffs and compounds characteristics.

    :param smiles: (List[str]) list of SMILES strings
    :param bioactivity: (List[float]) list of bioactivity values
    :param in_log10: (bool) are the bioactivity values in log10?
    :param n_clusters: (int) number of clusters the data is split into for getting homogeneous data splits
    :param test_size: (float) test split
    :param similarity:  (float) similarity threshold for calculating activity cliffs
    :param potency_fold: (float) potency difference threshold for calculating activity cliffs
    :param remove_stereo: (bool) Remove racemic mixtures altogether?

    :return: A tuple containing the train, test splits of the data.
    """
    assert sizes[2] == 0, "Current cliffs balanced split only supports train/val splits"
    smiles = data.smiles_train
    bioactivity = data.y_train
    test_size = sizes[1]

    if remove_stereo:
        stereo_smiles_idx = [smiles.index(i) for i in find_stereochemical_siblings(smiles)]
        smiles = [smi for i, smi in enumerate(smiles) if i not in stereo_smiles_idx]
        bioactivity = [act for i, act in enumerate(bioactivity) if i not in stereo_smiles_idx]
        if len(stereo_smiles_idx) > 0:
            print(f"Removed {len(stereo_smiles_idx)} stereoisomers")

    if not in_log10:
        # bioactivity = (10**abs(np.array(bioactivity))).tolist()
        bioactivity = (-np.log10(bioactivity)).tolist()
        log_data = data
        for d in log_data:
            d.y = -np.log10(d.y)
        data = log_data

    cliffs = ActivityCliffs(smiles, bioactivity)
    cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, similarity=similarity, potency_fold=potency_fold)

    # Perform spectral clustering on a tanimoto distance matrix
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=seed, affinity='precomputed')
    clusters = spectral.fit(get_tanimoto_matrix(smiles)).labels_

    train_idx, test_idx = [], []
    for cluster in range(n_clusters):

        cluster_idx = np.where(clusters == cluster)[0]
        clust_cliff_mols = [cliff_mols[i] for i in cluster_idx]

        # Can only split stratiefied on cliffs if there are at least 2 cliffs present, else do it randomly
        if sum(clust_cliff_mols) > 2:
            clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=test_size,
                                                               random_state=seed,
                                                               stratify=clust_cliff_mols, shuffle=True)
        else:
            clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=test_size,
                                                               random_state=seed,
                                                               shuffle=True)

        train_idx.extend(clust_train_idx)
        test_idx.extend(clust_test_idx)

    return MoleculeDataset(train_idx), MoleculeDataset(test_idx), 


def process_data(smiles: List[str], bioactivity: List[float], in_log10: bool = False, n_clusters: int = 5,
               test_size: float = 0.2, similarity: float = 0.9, potency_fold: int = 10, remove_stereo: bool = False):
    """ Split data into train/test according to activity cliffs and compounds characteristics.

    :param smiles: (List[str]) list of SMILES strings
    :param bioactivity: (List[float]) list of bioactivity values
    :param in_log10: (bool) are the bioactivity values in log10?
    :param n_clusters: (int) number of clusters the data is split into for getting homogeneous data splits
    :param test_size: (float) test split
    :param similarity:  (float) similarity threshold for calculating activity cliffs
    :param potency_fold: (float) potency difference threshold for calculating activity cliffs
    :param remove_stereo: (bool) Remove racemic mixtures altogether?

    :return: df[smiles, exp_mean [nM], y, cliff_mol, split]
    """
    return split_cliffs_balanced(smiles, bioactivity, in_log10, n_clusters, test_size, similarity,  potency_fold, remove_stereo)



def find_stereochemical_siblings(smiles: List[str]):
    """ Detects molecules that have different SMILES strings, but ecode for the same molecule with
    different stereochemistry. For racemic mixtures it is often unclear which one is measured/active

    Args:
        smiles: (lst) list of SMILES strings

    Returns: (lst) List of SMILES having a similar molecule with different stereochemistry

    """
    from cliffs_v0 import get_tanimoto_matrix

    lower = np.tril(get_tanimoto_matrix(smiles, radius=4, nBits=4096), k=0)
    identical = np.where(lower == 1)
    identical_pairs = [[smiles[identical[0][i]], smiles[identical[1][i]]] for i, j in enumerate(identical[0])]

    return list(set(sum(identical_pairs, [])))


def determine_atom_col(cp, 
                       atom_importance, 
                       norm,
                       cmap,
                       set_weights=False):
    """ Colors atoms with positive and negative contributions
    as green and red respectively, using an `eps` absolute
    threshold.

    Parameters
    ----------
    mol : rdkit mol
    atom_importance : np.ndarray
        importances given to each atom
    bond_importance : np.ndarray
        importances given to each bond
    version : int, optional
        1. does not consider bond importance
        2. bond importance is taken into account, but fixed
        3. bond importance is treated the same as atom importance, by default 2

    Returns
    -------
    dict
        atom indexes with their assigned color
    """
    atom_col = {}

    # Convert importance scores to colors
    colors = [cmap(norm(score))[:3] for score in atom_importance]

    for idx, v in enumerate(zip(atom_importance, colors)):
        v, c = v
        atom_col[idx] = c
        if set_weights:
            cp.GetAtomWithIdx(idx).SetProp("atomNote","%.3f"%(v))
    return atom_col, cp

from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from utils import determine_atom_col
from IPython.display import SVG

def visualization(smile, atom_imp, norm,  set_weights=False, svg_dir=None, vis_factor=1.0, img_width=400, img_height=200, testing=True, training=False, drawAtomIndices=False):
    svg_list = []
    #for idx, smiles in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smile)
    cp = Chem.Mol(mol)

    num_atoms = mol.GetNumAtoms()
    assert num_atoms == len(atom_imp), "Number of atoms in molecule and atom importance vector do not match."
    
    highlightAtomColors, cp = determine_atom_col(cp, atom_imp, eps=threshold, use_negative=use_negative, set_weights=set_weights)
    highlightAtoms = list(highlightAtomColors.keys())
    #highlightBondColors = determine_bond_col(highlightAtomColors, mol)
    #highlightBonds = list(highlightBondColors.keys())

    highlightAtomRadii = {
        # k: np.abs(v) * vis_factor for k, v in enumerate(atom_imp)
        k: 0.1 * vis_factor for k, v in enumerate(atom_imp)
    }

    rdDepictor.Compute2DCoords(cp, canonOrient=True)
    #drawer = rdMolDraw2D.MolDraw2DCairo(img_width, img_height)
    drawer = rdMolDraw2D.MolDraw2DSVG(img_width, img_height)
    if drawAtomIndices:
        drawer.drawOptions().addAtomIndices = True
    drawer.drawOptions().useBWAtomPalette()
    drawer.DrawMolecule(
        cp,
        highlightAtoms=highlightAtoms,
        highlightAtomColors=highlightAtomColors,
        # highlightAtomRadii=highlightAtomRadii,
        #highlightBonds=highlightBonds,
        #highlightBondColors=highlightBondColors,
    )
    drawer.FinishDrawing()
    #drawer.WriteDrawingText(os.path.join(svg_dir, f"{smiles_idx_list[idx]}.png"))
    svg = drawer.GetDrawingText()#.replace("svg:", "")
    #svg_list.append(svg)

    #return svg_list
    return svg