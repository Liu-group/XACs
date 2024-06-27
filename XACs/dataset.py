import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import torch
from XACs.featurization import MolTensorizer
import os
import random
from typing import List, Union
from copy import deepcopy
from torch_geometric.data import Batch, Data
from XACs.cliffs import ActivityCliffs, get_tanimoto_matrix
from XACs.utils import DATASETS, MOLDATASETS
from typing import Optional
from sklearn.model_selection import train_test_split


class MoleculeDataset:
    def __init__(
        self, 
        file: str, 
        working_dir: str = None):
        """ 
        Data class to easily load featurized molecular data, including activity cliff information
        """

        if os.path.exists(file):
            df = pd.read_csv(file)     
        else:
            self.dataset_name = file
            if self.dataset_name in DATASETS or self.dataset_name in MOLDATASETS:
                assert working_dir is not None, "Please specify a working directory"
                self.working_path = os.path.join(working_dir, self.dataset_name)
                file = os.path.join(self.working_path, f"{self.dataset_name}.csv")
                print(f"Loading dataset {self.dataset_name} from {file}")
                df = pd.read_csv(file)
            else:
                print(f"Dataset {self.dataset_name} not found in {working_dir}")

        self.smiles_all = df['smiles'].tolist()
        self.y_all = df['y'].tolist()
        self.cliff_mols = None

        self.featurize_data()
        
    def get_cliffs(self, sim_thre: float = 0.9, dist_thre: float = 1.0):
        # get cliff info
        dict_path = os.path.join(self.working_path, f'mcs_dict_{sim_thre}.pkl' if dist_thre==1.0 else f'mcs_dict_{sim_thre}_{dist_thre}.pkl')
        self.cliff = ActivityCliffs(self.smiles_all, 
                                    self.y_all,
                                    sim_thre=sim_thre, 
                                    dist_thre=dist_thre, 
                                    dict_path=dict_path)
        self.cliff_mols = self.cliff.cliff_mols      
        self.cliff_dict = self.cliff.mcs_dict
        for i in range(len(self.data_all)):
            self.data_all[i].cliff = self.cliff_mols[i]

    def split_data(self,
                split_ratio: List[float] = [0.8, 0.1, 0.1],
                split_method: str = 'random',
                n_clusters: Optional[int] = 5,
                seed: int = 42,
                save_split: bool = False,
                return_idx: bool = False):

        ratio = "".join([str(int(r*10)) for r in split_ratio])
        split_path = os.path.join(self.working_path, f"{self.dataset_name}_{split_method}_{ratio}_{seed}.csv")
        if os.path.exists(split_path):
            df = pd.read_csv(split_path)
            train_idx, val_idx, test_idx = df[df['split'] == 'train'].index.tolist(), df[df['split'] == 'val'].index.tolist(), df[df['split'] == 'test'].index.tolist()

        if split_method == 'random':
            train_idx, test_idx = train_test_split(range(len(self.smiles_all)), test_size=split_ratio[2], random_state=seed)
            train_idx, val_idx = train_test_split(train_idx, test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]), random_state=seed)
        elif split_method == 'cliff':
            assert self.cliff_mols is not None, "No cliff information available"
            train_idx, val_idx, test_idx = cliff_split(self.smiles_all, self.y_all, self.cliff_mols, split_ratio=split_ratio, n_clusters=n_clusters, seed=seed)
        else:
            raise ValueError(f"Split method {split_method} not recognized")  

        if save_split:        
            split = []
            for i in range(len(self.smiles_all)):
                if i in train_idx:
                    split.append('train')
                elif i in val_idx:
                    split.append('val')
                elif i in test_idx:
                    split.append('test')
                else:
                    raise ValueError(f"Can't find molecule {i} in train, val or test")
            df = pd.DataFrame({'smiles': self.smiles_all,
                            'y': self.y_all,
                            'cliff_mol': self.cliff_mols,
                            'split': split})
            df.to_csv(split_path, index=False)
            print(f"Saved split to {split_path}")
    
        if return_idx == True:
            return train_idx, val_idx, test_idx
        else:
            data_train, data_val, data_test = [self.data_all[i] for i in train_idx], [self.data_all[i] for i in val_idx], [self.data_all[i] for i in test_idx]
            return data_train, data_val, data_test

    def featurize_data(self):
        featurizer = MolTensorizer()
        self.data_all = [featurizer.tensorize(smi) for smi in tqdm(self.smiles_all)]
        self.num_node_features = self.data_all[0].num_node_features
        self.num_edge_features = self.data_all[0].num_edge_features
        # concatenate data with smiles, target and whether or not cliff_mol.
        for i in range(len(self.data_all)):
            self.data_all[i].smiles = self.smiles_all[i]
            self.data_all[i].target = self.y_all[i]
            if self.cliff_mols is not None:
                self.data_all[i].cliff = self.cliff_mols[i]

def pack_data(data: Data, cliff_dict: dict, space: Optional[Data] = None) -> Data:
    """
    Shape data.x from (num_node_of_mol_i, num_node_features) to (sum(num_node_of_mol_i, num_node_of_cliff_mol), num_node_features);
    data.edge_index and data.edge_attr are transformed into several disconnected graphs using Batch;

    atom_mask = [atom_mask_i, atom_mask_j] with shape (max_num_cliff_pairs_in_list, num_atom_i)
    e.g. 
    max_num_cliff_pairs_in_list = 3;
    molecule_i has 2 cliff pair: molecule_j and molecule_k;
    uncom_atom_mask_i = [[1, 1, 1, 0, 0, 0],
                         [0, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0]] 
    com_atom_mask_i = [[0, 0, 0, 1, 1, 1],
                       [1, 0, 0, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0]]
    meaning the first 3 atoms are the target attribution substrucutre of molecule i corresponding to the first cliff pair,
    the 2rd and 3rd atoms are the target attribution substrucutre of molecule i corresponding to the second cliff pair;

    uncom_atom_mask_j = [[-1, 0, 0, 0],
                        [  0, 0, 0, 0],
                        [  0, 0, 0, 0]]
    common_atom_mask_j = [[ 0, -1,  -1, -1],
                           [0,  0,   0,  0],
                           [0,  0,   0,  0]]

    uncom_atom_mask_k = [[0,  0, 0, 0,  0, 0],
                        [ 0, -1, 0, 0, -1, 0],
                        [ 0,  0, 0, 0,  0, 0]]
    common_atom_mask_k = [[0,  0, 0, 0,  0, 0],
                        [-1,  0, -1, -1,  0, -1],
                        [0,  0, 0, 0,  0, 0]]
    meaning the first 3 atoms are the target attribution substrucutre of molecule j;
    the 2rd and 5th atoms are the target attribution substrucutre of molecule k;
    uncom_atom_mask = [[1, 1, 1, 0, 0, 0, -1,  0,  0, 0, 0,  0, 0, 0,  0, 0],
                       [0, 1, 1, 0, 0, 0,  0,  0,  0, 0, 0, -1, 0, 0, -1, 0],
                       [0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0,  0, 0, 0,  0, 0]] 
    common_atom_mask = [[0, 0, 0, 1, 1, 1,  0, -1, -1,-1,  0, 0, 0, 0,  0,  0],
                        [1, 0, 0, 1, 1, 1,  0,  0,  0, 0, -1, 0,-1,-1,  0, -1],
                        [0, 0, 0, 0, 0, 0,  0,  0,  0, 0,  0, 0, 0, 0,  0,  0]]
    """
    # if search space is not provided, only search for mmps present in the given smiles list
    if space != None:
        smiles_all = [space[i].smiles for i in range(len(space))]
        data_all = space
    else:
        smiles_all = [data[i].smiles for i in range(len(data))]
        data_all = data

    smiles = [data[i].smiles for i in range(len(data))]
    packed_data = deepcopy(data)

    # Iterate through the values in the dictionary to check the maximum number of mmp for one molecule
    max_length = 1
    for value in cliff_dict.values():
        if len(value) > max_length:
            max_length = len(value) - 1

    for i in range(len(packed_data)):
        smiles_i = str(smiles[i])
        num_atom_i = packed_data[i].x.size(0)
        # get the valid mmps that are in the train smiles list
        mmps = cliff_dict[smiles_i][1:]
        available_mmps = [mmp_dict for mmp_dict in mmps if mmp_dict['smiles'] in smiles_all]
        num_av_mmp = len(available_mmps)

        potency_diff = torch.zeros(max_length, 1)
        if num_av_mmp == 0:
            uncom_atom_mask = torch.zeros(max_length, num_atom_i)
            common_atom_mask = torch.zeros(max_length, num_atom_i)
            packed_data[i].mini_batch = torch.zeros(num_atom_i).long()
        else:
            mmp_data_list = [packed_data[i]]
            uncom_atom_mask_i = torch.zeros(max_length, num_atom_i)
            common_atom_mask_i = torch.zeros(max_length, num_atom_i)
            for j, mmp in enumerate(available_mmps):

                uncom_atom_mask_i[j] = torch.tensor(mmp['atom_mask_i']).reshape(1, num_atom_i)
                common_atom_mask_i[j] = torch.where(uncom_atom_mask_i[j] == 0, 1, 0)
                assert torch.equal(uncom_atom_mask_i[j] + common_atom_mask_i[j], torch.ones(num_atom_i)), \
                        "uncom_atom_mask_i + common_atom_mask_i != torch.ones(1, num_atom_i)"
            uncom_atom_mask = uncom_atom_mask_i
            common_atom_mask = common_atom_mask_i
            for j, mmp in enumerate(available_mmps):
                j_idx = smiles_all.index(mmp['smiles'])
                num_atom_j = len(mmp['atom_mask_j'])

                uncom_atom_mask_j = torch.zeros(max_length, num_atom_j)
                uncom_atom_mask_j[j] = -1.0*torch.tensor(mmp['atom_mask_j']).reshape(1, num_atom_j)
                common_atom_mask_j = torch.zeros(max_length, num_atom_j)
                common_atom_mask_j[j] = -1.0*torch.where(uncom_atom_mask_j[j] == 0, 1, 0)

                assert torch.equal(uncom_atom_mask_j[j] + common_atom_mask_j[j], -1.0*torch.ones(num_atom_j)), "uncom_atom_mask_j + common_atom_mask_j != -1.0*torch.ones(1, num_atom_j)"

                uncom_atom_mask = torch.cat([uncom_atom_mask, uncom_atom_mask_j], dim=-1)
                common_atom_mask = torch.cat([common_atom_mask, common_atom_mask_j], dim=-1)
                potency_diff[j] = float(mmp['potency_diff'])      
                # concatenate x, edge_index, edge_attr
                mmp_data_list.append(data_all[j_idx])

            batched_data = Batch.from_data_list(mmp_data_list)
            packed_data[i].x, packed_data[i].edge_index, packed_data[i].edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
            packed_data[i].mini_batch = batched_data.batch
        graph_mask = torch.zeros(num_av_mmp + 1)
        graph_mask[0] = 1.
        packed_data[i].graph_mask = graph_mask
        packed_data[i].potency_diff = potency_diff.T
        packed_data[i].uncom_atom_mask = uncom_atom_mask.T 
        packed_data[i].common_atom_mask = common_atom_mask.T
        packed_data[i].smiles = [smiles_i] + [mmp['smiles'] for mmp in available_mmps]
    return packed_data

def cliff_split(smiles_all,
                y_all,
                cliff_mols,
                split_ratio: List[float] = [0.8, 0.1, 0.1],
                n_clusters: int = 5, 
                seed: int = 42):
        """
        Split data into train/val/test according to activity cliffs.
        """
        from sklearn.cluster import SpectralClustering

        # Perform spectral clustering on a tanimoto distance matrix
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=seed, affinity='precomputed')
        clusters = spectral.fit(get_tanimoto_matrix(smiles_all)).labels_
        train_idx, val_idx, test_idx = [], [], []
        for cluster in range(n_clusters):
                cluster_idx = np.where(clusters == cluster)[0]
                clust_cliff_mols = [cliff_mols[i] for i in cluster_idx]
                # Can only split stratiefied on cliffs if there are at least 3 cliffs present, else do it randomly
                if sum(clust_cliff_mols) > 3:
                    clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=split_ratio[2],
                                                                    random_state=seed,
                                                                    stratify=clust_cliff_mols, shuffle=True)
                    clust_train_idx, clust_val_idx = train_test_split(clust_train_idx, test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]),
                                                                    random_state=seed,
                                                                    stratify=[cliff_mols[i] for i in clust_train_idx], shuffle=True)
                else:
                    clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=split_ratio[2],
                                                                    random_state=seed,
                                                                    shuffle=True)
                    clust_train_idx, clust_val_idx = train_test_split(clust_train_idx, test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]),
                                                                    random_state=seed,
                                                                    shuffle=True)
    
                train_idx.extend(clust_train_idx)
                val_idx.extend(clust_val_idx)
                test_idx.extend(clust_test_idx)

        return train_idx, val_idx, test_idx






    

    

    
