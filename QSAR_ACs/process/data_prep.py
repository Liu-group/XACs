"""
Author: Derek van Tilborg -- TU/e -- 22-05-2022

A collection of data-prepping functions
    - split_data():             split ChEMBL csv into train/test taking similarity and cliffs into account. If you want
                                to process your own data, use this function
    - process_data():           see split_data()
    - load_data():              load a pre-processed dataset from the benchmark
"""

from cliffs_v0 import ActivityCliffs, get_tanimoto_matrix
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from typing import List
import pandas as pd
import numpy as np

RANDOM_SEED = 42


def split_data(smiles: List[str], bioactivity: List[float], in_log10: bool = False, n_clusters: int = 5,
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


    if not in_log10:
        # bioactivity = (10**abs(np.array(bioactivity))).tolist()
        bioactivity = (-np.log10(bioactivity)).tolist()

    cliffs = ActivityCliffs(smiles, bioactivity)
    cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, similarity=similarity, potency_fold=potency_fold)

    # Perform spectral clustering on a tanimoto distance matrix
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=RANDOM_SEED, affinity='precomputed')
    clusters = spectral.fit(get_tanimoto_matrix(smiles)).labels_

    train_idx, test_idx = [], []
    for cluster in range(n_clusters):

        cluster_idx = np.where(clusters == cluster)[0]
        clust_cliff_mols = [cliff_mols[i] for i in cluster_idx]

        # Can only split stratiefied on cliffs if there are at least 2 cliffs present, else do it randomly
        if sum(clust_cliff_mols) > 2:
            clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=test_size,
                                                               random_state=RANDOM_SEED,
                                                               stratify=clust_cliff_mols, shuffle=True)
        else:
            clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=test_size,
                                                               random_state=RANDOM_SEED,
                                                               shuffle=True)

        train_idx.extend(clust_train_idx)
        test_idx.extend(clust_test_idx)

    train_test = []
    for i in range(len(smiles)):
        if i in train_idx:
            train_test.append('train')
        elif i in test_idx:
            train_test.append('test')
        else:
            raise ValueError(f"Can't find molecule {i} in train or test")

    return pd.DataFrame({'smiles': smiles,
                         'exp_mean [nM]': (10**(np.array(bioactivity)*-1)).tolist(),
                         'y': bioactivity,
                         'cliff_mol': cliff_mols,
                         'split': train_test})


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
    return split_data(smiles, bioactivity, in_log10, n_clusters, test_size, similarity,  potency_fold, remove_stereo)



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

if __name__ == '__main__':
    ## This is th script that will be used to process the data from the paper
    ## "https://doi.org/10.1186/s13321-023-00708-w";
    ## git: "https://github.com/MarkusFerdinandDablander/QSAR-activity-cliff-experiments"

    # get name from command line
    name = 'postera_sars_cov_2_mpro'
    smiles = pd.read_csv('../'+name+'/molecule_data_clean.csv')['SMILES'].tolist()
    try:
        bioactivity = pd.read_csv('../'+name+'/molecule_data_clean.csv')['Ki [nM]'].tolist()
    except:
        bioactivity = pd.read_csv('../'+name+'/molecule_data_clean.csv')['f_avg_IC50 [uM]'].tolist()
    pd_processed = split_data(smiles=smiles, bioactivity=bioactivity)
    # save the processed data
    pd_processed.to_csv('../'+name+'/processed_data.csv', index=False)
    print(f'{name} processing done')