import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from featurization import MolTensorizer
import os
import random
from typing import List, Union
from cliffs import ActivityCliffs
from copy import deepcopy
from torch_geometric.data import Batch, Data
from sklearn.model_selection import train_test_split
from sklearn.cluster import SpectralClustering
from cliffs import ActivityCliffs, get_tanimoto_matrix

#DATASETS = ['chembl_dopamine_d2', 'CHEMBL4203_Ki', 'chembl_factor_xa', 'postera_sars_cov_2_mpro' ]
DATASETS = ['CHEMBL4616_EC50',  'CHEMBL4792_Ki', 'CHEMBL1871_Ki', 'CHEMBL2971_Ki',
 'CHEMBL239_EC50', 'CHEMBL233_Ki','CHEMBL235_EC50', 'CHEMBL231_Ki','CHEMBL218_EC50','CHEMBL244_Ki',
 'CHEMBL234_Ki','CHEMBL237_Ki','CHEMBL1862_Ki','CHEMBL4203_Ki','CHEMBL2047_EC50',
 'CHEMBL219_Ki','CHEMBL236_Ki','CHEMBL228_Ki','CHEMBL2147_Ki','CHEMBL204_Ki',
 'CHEMBL262_Ki','CHEMBL287_Ki','CHEMBL2034_Ki','CHEMBL3979_EC50','CHEMBL238_Ki','CHEMBL2835_Ki',
 'CHEMBL4005_Ki','CHEMBL237_EC50','CHEMBL264_Ki','CHEMBL214_Ki']#,'postera_sars_cov_2_mpro','chembl_factor_xa','chembl_dopamine_d2',]
WORKING_DIR = 'QSAR_ACs'

class MoleculeDataset:
    def __init__(self, file: str):
        """ 
        Data class to easily load featurized molecular bioactivity data, including activity cliff information
        """

        if os.path.exists(file):
            df = pd.read_csv(file)     
        else:
            self.dataset_name = file
            if self.dataset_name in DATASETS:
                self.working_path = os.path.join(WORKING_DIR, self.dataset_name)
                file = os.path.join(self.working_path, f"{self.dataset_name}.csv")
                print(f"Loading dataset {self.dataset_name} from {file}")
                df = pd.read_csv(file)
            else:
                print(f"Dataset {self.dataset_name} not found in {WORKING_DIR}")

        self.smiles_all = df['smiles'].tolist()
        self.y_all = df['y'].tolist()
        self.featurize_data()

    def cliff_split_train_test(self, 
                               test_size: float = 0.2, 
                               n_clusters: int = 5, 
                               seed: int = 42, 
                               threshold: float = 0.9,
                               save_split: bool = False) -> List[int]:
        """ 
        Split data into train/test according to activity cliffs.
        """
        # Perform spectral clustering on a tanimoto distance matrix
        spectral = SpectralClustering(n_clusters=n_clusters, random_state=seed, affinity='precomputed')
        clusters = spectral.fit(get_tanimoto_matrix(self.smiles_all)).labels_

        train_idx, test_idx = [], []
        for cluster in range(n_clusters):

            cluster_idx = np.where(clusters == cluster)[0]
            clust_cliff_mols = [self.cliff_mols[i] for i in cluster_idx]

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

        train_test = []
        for i in range(len(self.smiles_all)):
            if i in train_idx:
                train_test.append('train')
            elif i in test_idx:
                train_test.append('test')
            else:
                raise ValueError(f"Can't find molecule {i} in train or test")
        
        if save_split:
            df = pd.DataFrame({'smiles': self.smiles_all,
                            'exp_mean [nM]': (10**(np.array(self.y_all)*-1)).tolist(),
                            'y': self.y_all,
                            'cliff_mol': self.cliff_mols,
                            'split': train_test})
            df.to_csv(os.path.join(self.working_path, f"{self.dataset_name}_cliff_thre_{threshold}_split_{seed}.csv"), index=False)
            print(f"Saved split to {self.working_path}")
        return train_idx, test_idx

    def featurize_data(self):
        featurizer = MolTensorizer()
        self.data_all = [featurizer.tensorize(smi) for smi in tqdm(self.smiles_all)]
        self.num_node_features = self.data_all[0].num_node_features
        self.num_edge_features = self.data_all[0].num_edge_features
    
    def get_simple_cliff_labels(self):
        label_dir = os.path.join(WORKING_DIR, self.dataset_name, 'att_train.pt') 
        ac_train = ActivityCliffs(self.smiles_train, self.y_train, label_dir=label_dir, dataset_name=self.dataset_name)
        #ac.find_cliffs()
        att_train = ac_train.get_all_cliff_attributions()
        self.att_train = [torch.tensor(np.array(att)) for att in att_train]
    
    def conca_data(self):
        # concatenate data with smiles, target and whether or not cliff_mol.
        for i in range(len(self.data_all)):
            self.data_all[i].smiles = self.smiles_all[i]
            self.data_all[i].target = self.y_all[i]
            self.data_all[i].cliff = self.cliff_mols[i]

    def conca_data_cliff(self):
        batched_data_train = deepcopy(self.x_train)
        max_length = 1
        # Iterate through the values in the dictionary to check the maximum number of mmp for one molecule
        for value in self.cliff_dict.values():
            if len(value) > max_length:
                max_length = len(value) - 1
        for i in range(len(batched_data_train)):
            smiles = str(self.smiles_train[i])
            num_atom_i = batched_data_train[i].x.size(0)
            mmps = self.cliff_dict[smiles][1:]
            # get the valid mmps that are in the train smiles list
            available_mmps = [mmp_dict for mmp_dict in mmps if mmp_dict['smiles'] in self.smiles_train]
            num_av_mmp = len(available_mmps)
            potency_diff = torch.zeros(max_length, 1)
            if num_av_mmp == 0:
                atom_mask = torch.zeros(max_length, num_atom_i)
                batched_data_train[i].x, batched_data_train[i].edge_index, batched_data_train[i].edge_attr = batched_data_train[i].x, batched_data_train[i].edge_index, batched_data_train[i].edge_attr
            else:
                mmp_data_list = [batched_data_train[i]]
                atom_mask_i = torch.zeros(max_length, num_atom_i)
                for j, mmp in enumerate(available_mmps):
                    atom_mask_i[j] = torch.tensor(mmp['atom_mask_i']).reshape(1, num_atom_i)
                atom_mask = atom_mask_i
                for j, mmp in enumerate(available_mmps):
                    j_idx = self.smiles_train.index(mmp['smiles'])
                    num_atom_j = len(mmp['atom_mask_j'])
                    atom_mask_j = torch.zeros(max_length, num_atom_j)
                    atom_mask_j[j] = -1.0*torch.tensor(mmp['atom_mask_j']).reshape(1, num_atom_j)
                    atom_mask = torch.cat([atom_mask, atom_mask_j], dim=-1)
                    potency_diff[j] = float(mmp['potency_diff'])      
                    # concatenate x, edge_index, edge_attr
                    mmp_data_list.append(self.x_train[j_idx])

                #targ = torch.tensor([np.nan]*(num_av_mmp + 1))
                #targ[0] = self.y_train[i]
                #data_train[i].target = targ.reshape(-1,1)
                batched_data = Batch.from_data_list(mmp_data_list)
                batched_data_train[i].x, batched_data_train[i].edge_index, batched_data_train[i].edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
            batched_data_train[i].potency_diff = potency_diff
            batched_data_train[i].atom_mask = atom_mask.T 
            batched_data_train[i].target = torch.tensor(self.y_train[i]).reshape(-1,1)   
            # make the att that is all 0 or 1 to nan
            # and reshape the nan att to [node,1] to avoid error for later batch collate
            '''data_train[i].att_label = torch.tensor([np.nan]*data_train[i].x.size(0)).reshape(-1,1) \
                                if torch.isnan(self.att_train[i]).any() \
                                or self.att_train[i].sum()==0 \
                                or self.att_train[i].sum()==len(self.att_train[i]) \
                                else self.att_train[i].reshape(-1,1)'''
            batched_data_train[i].cliff = self.cliff_mols_train[i]
            batched_data_train[i].smiles = [smiles] + [mmp['smiles'] for mmp in available_mmps]
            batched_data_train[i].pred_mask = torch.cat([torch.ones(num_atom_i), torch.zeros(batched_data_train[i].atom_mask.size(0)-num_atom_i)]).reshape(-1,1)
        self.batched_data_train = batched_data_train    
        '''data_test = deepcopy(self.x_test)
        for i in range(len(data_test)):
            data_test[i].smiles = self.smiles_test[i]
            data_test[i].target = self.y_test[i]
            data_test[i].cliff = self.cliff_mols_test[i]

        self.data_test = data_test'''      

    def shuffle(self):
        """ Shuffle training data """
        c = list(zip(self.x_train, self.smiles_train, self.y_train, self.cliff_mols_train))  # Shuffle all lists together
        random.shuffle(c)
        self.x_train, self.smiles_train, self.y_train, self.cliff_mols_train = zip(*c)

        self.x_train = list(self.x_train)
        self.smiles_train = list(self.smiles_train)
        self.y_train = list(self.y_train)
        self.cliff_mols_train = list(self.cliff_mols_train)

    def __call__(self, 
                 concat: bool = True,
                 seed: int = 42, 
                 threshold: float = 0.9,
                 save_split: bool = True):
        dict_path = os.path.join(WORKING_DIR, self.dataset_name, f'mcs_dict_{threshold}_copy.pkl')
        split_path = os.path.join(self.working_path, f"{self.dataset_name}_cliff_thre_{threshold}_split_{seed}.csv")
        
        self.cliff = ActivityCliffs(self.smiles_all, self.y_all, threshold=threshold, dict_path=dict_path)
        self.cliff_mols = self.cliff.cliff_mols
        self.conca_data()
        if os.path.exists(split_path):
            df = pd.read_csv(split_path)
            train_idx, test_idx = df[df['split'] == 'train'].index.tolist(), df[df['split'] == 'test'].index.tolist() 
        else:   
            self.cliff_mols = self.cliff.cliff_mols    
            train_idx, test_idx = self.cliff_split_train_test(
                                            seed = seed, 
                                            threshold = threshold,
                                            save_split = save_split)
            
        data_train = [self.data_all[i] for i in train_idx]
        data_test = [self.data_all[i] for i in test_idx]
        self.cliff_dict = deepcopy(self.cliff.mcs_dict)
        self.batched_data_train = pack_data(data_train, self.cliff_dict)
        self.data_train = data_train
        # mols in test set is searched within the whole dataset for cliff pairs
        data_all = pack_data(self.data_all, self.cliff_dict)
        self.batched_data_test = [data_all[i] for i in test_idx]
        self.data_test = [self.data_all[i] for i in test_idx]

    def __repr__(self):
        return f"Data object with molecules as: {len(self.y_train)} train/{len(self.y_test)} test"
    

def get_test_cliff(y_test_pred: Union[List[float], np.array],
                    y_test: Union[List[float], np.array],
                    cliff_mols_test: List[int] = None):
    """ get the testset of activity cliff compounds

    :param y_test_pred: (lst/array) predicted test values
    :param y_test: (lst/array) true test values
    :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
    :return: float testset of activity cliff compounds
    """

    # Convert to numpy array if it is not
    y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
    y_test = np.array(y_test) if type(y_test) is not np.array else y_test

    # Get the index of the activity cliff molecules
    cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

    # Filter out only the predicted and true values of the activity cliff molecules
    y_pred_cliff_mols = y_test_pred[cliff_test_idx]
    y_test_cliff_mols = y_test[cliff_test_idx]

    return y_pred_cliff_mols, y_test_cliff_mols

def pack_data(data: Data, cliff_dict: dict) -> Data:
    """
    Shape data.x from (num_node_of_mol_i, num_node_features) to (sum(num_node_of_mol_i, num_node_of_cliff_mol), num_node_features);
    data.edge_index and data.edge_attr are transformed into several disconnected graphs using Batch;

    atom_mask = [atom_mask_i, atom_mask_j] with shape (max_num_cliff_pairs_in_list, num_atom_i)
    e.g. 
    max_num_cliff_pairs_in_list = 3;
    molecule_i has 2 cliff pair: molecule_j and molecule_k;
    atom_mask_i = [[1, 1, 1, 0, 0, 0],
                   [0, 1, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0]] 
    meaning the first 3 atoms are the target attribution substrucutre of molecule i corresponding to the first cliff pair,
    the 2rd and 3rd atoms are the target attribution substrucutre of molecule i corresponding to the second cliff pair;
    atom_mask_j = [[-1, -1, -1, 0],
                   [ 0,  0,  0, 0],
                   [ 0,  0,  0, 0]]
    atom_mask_k = [[0,  0, 0, 0,  0, 0],
                   [0, -1, 0, 0, -1, 0],
                   [0,  0, 0, 0,  0, 0]]
    meaning the first 3 atoms are the target attribution substrucutre of molecule j;
    the 2rd and 5th atoms are the target attribution substrucutre of molecule k;
    atom_mask = [[1, 1, 1, 0, 0, 0, -1, -1, -1, 0, 0,  0, 0, 0,  0, 0],
                 [0, 1, 1, 0, 0, 0,  0,  0,  0, 0, 0, -1, 0, 0, -1, 0],
                 [0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0,  0, 0, 0,  0, 0]] 
    """
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
        available_mmps = [mmp_dict for mmp_dict in mmps if mmp_dict['smiles'] in smiles]
        num_av_mmp = len(available_mmps)

        potency_diff = torch.zeros(max_length, 1)
        if num_av_mmp == 0:
            atom_mask = torch.zeros(max_length, num_atom_i)
            packed_data[i].mini_batch = torch.zeros(num_atom_i).long()
        else:
            mmp_data_list = [packed_data[i]]
            atom_mask_i = torch.zeros(max_length, num_atom_i)
            for j, mmp in enumerate(available_mmps):
                atom_mask_i[j] = torch.tensor(mmp['atom_mask_i']).reshape(1, num_atom_i)
            atom_mask = atom_mask_i
            for j, mmp in enumerate(available_mmps):
                j_idx = smiles.index(mmp['smiles'])
                num_atom_j = len(mmp['atom_mask_j'])
                atom_mask_j = torch.zeros(max_length, num_atom_j)
                atom_mask_j[j] = -1.0*torch.tensor(mmp['atom_mask_j']).reshape(1, num_atom_j)
                atom_mask = torch.cat([atom_mask, atom_mask_j], dim=-1)
                potency_diff[j] = float(mmp['potency_diff'])      
                # concatenate x, edge_index, edge_attr
                mmp_data_list.append(data[j_idx])

            batched_data = Batch.from_data_list(mmp_data_list)
            packed_data[i].x, packed_data[i].edge_index, packed_data[i].edge_attr = batched_data.x, batched_data.edge_index, batched_data.edge_attr
            packed_data[i].mini_batch = batched_data.batch
        packed_data[i].potency_diff = potency_diff
        packed_data[i].atom_mask = atom_mask.T 
        packed_data[i].smiles = [smiles_i] + [mmp['smiles'] for mmp in available_mmps]
    return packed_data




    

    

    
