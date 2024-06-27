
import pandas as pd
import json
import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data)
from rdkit import Chem
from featurization import MolTensorizer
def load_human_data(path):
    """
    Load human annotation data for a dataset

    input:  config contains base folder path to all the human masks,
            dataset name

    output: dict file with format:

            train:
                img_idx:
                    node_importance:
                    edge_importance:
            val:
                img_idx:
                    node_importance:
                    edge_importance:
            test:
                img_idx:
                    node_importance:
                    edge_importance:
    """
    base_fp = path
    dataset = 'BBBP'
    # only load the human mask for training if 'human_mask' is True
    train_fp = os.path.join(base_fp, dataset+'_train.csv')
    train_data = pd.read_csv(train_fp)
    train_dict = read_human_data_from_pd(train_data)


    val_fp = os.path.join(base_fp, dataset+'_val.csv')
    val_data = pd.read_csv(val_fp)
    val_dict= read_human_data_from_pd (val_data)

    test_fp = os.path.join(base_fp, dataset+'_test.csv')
    test_data = pd.read_csv(test_fp)
    test_dict = read_human_data_from_pd (test_data)

    return {"train": train_dict,
            "val": val_dict,
            "test":test_dict}

def read_human_data_from_pd(data):
    dict = {}
    N = len(data)
    for i in range(N):

        skip_flag = data["status"][i]

        if skip_flag == "labeled":
            index = data["img_idx"][i]

            if index not in dict:
                dict[index]={}
            else:
                print('duplication detected in human mask for img_idx:', index)

            human_mask = json.loads(data["record"][i])
            dict[index]['node_importance'] = human_mask["node_importance"]
            dict[index]['edge_importance'] = human_mask["edge_importance"]

    return dict


class BBBPDataset(InMemoryDataset):
    def __init__(self, 
                 root, 
                 name, 
                 featurization,
                 transform=None, 
                 pre_transform=None, 
                 pre_filter=None):
        #self.name = name.lower()
        self.name = name
        self.featurization = featurization
        super().__init__(root, transform, pre_transform, pre_filter)
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')
    
    @property
    def processed_dir(self):
        return osp.join(self.root, self.name, 'processed')
    
    @property
    def raw_file_names(self):
        return f'{self.name}.csv'

    @property
    def processed_file_names(self):
        if self.featurization== 'simplified':
            return 'BBBP_data_sim.pt'
        else:
            return 'BBBP_data.pt'
    
    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smis = df['smiles'].values.tolist()
        mols = [Chem.MolFromSmiles(smi) for smi in smis]

        ys = df['p_np'].values.tolist()
        ys = torch.tensor(ys, dtype=torch.float).view(-1, 1)

        data_list = []
        idx = 1
        for mol, y, smi in tqdm(zip(mols, ys, smis)):
            if mol is None:
                print(f'{idx} mol is None')
                continue
            tensorizer = MolTensorizer(featurization=self.featurization)
            x, e_idx, e_att = tensorizer.tensorize(mol)
            # Convert to graph
            data = Data(x=x, edge_index=e_idx, e=e_att, y=y, smile=smi, idx=idx)
            data_list.append(data)
            #if self.pre_filter is not None and not self.pre_filter(data):
                #continue
        
            if self.pre_transform is not None:
                data = self.pre_transform(data)
        print(len(data_list))
        #torch.save(self.collate(sv_data_list), osp.join(self.processed_dir, f'solvent_data.pt'))
        #torch.save(self.collate(sl_data_list), osp.join(self.processed_dir, f'solute_data.pt'))
        if self.featurization== 'simplified':
            torch.save(data_list, osp.join(self.processed_dir, f'BBBP_data_sim.pt'))

        else:
            torch.save(data_list, osp.join(self.processed_dir, f'BBBP_data.pt'))
        idx += 1
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if self.featurization== 'simplified':
            data = torch.load(osp.join(self.processed_dir, f'BBBP_data_sim.pt'))
        else:
            data = torch.load(osp.join(self.processed_dir, f'BBBP_data.pt'))
        return data