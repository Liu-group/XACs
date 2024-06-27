# Description: calculate the steric effect of each dataset
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from chembl_webresource_client.new_client import new_client
from dataset import MoleculeDataset
import pandas as pd
from const import DATASETS
DATA_PATH = 'QSAR_ACs'
molecule = new_client.molecule

def calculate_molecular_volume(chembl_id, molecule_smiles):
    m = molecule.filter(chembl_id=str(chembl_id)).only(['molecule_chembl_id', 'pref_name', 'molecule_structures'])
    if len(m) == 0:
        print(f'No molecule found for {chembl_id}, try to generate 3D structure from smiles')
        try:
            mol = Chem.MolFromSmiles(molecule_smiles)
            # add hydrogens
            mol = Chem.AddHs(mol)
            if mol is not None:
                # Compute 3D coordinates
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol, useRandomCoords=True)
        except: 
            print(f'Cannot generate 3D structure for {chembl_id}')
            return 0.
    else:
        mol = Chem.rdmolfiles.MolFromMolBlock(m[0]['molecule_structures']['molfile'])
    volume = AllChem.ComputeMolVolume(mol)
    return volume


# get cliff_dict for each dataset
def cal_steric_effect(dataset_name):        
    df = pd.read_csv(f'./Benchmark_Data/{dataset_name}/{dataset_name}.csv')
    chembl_ids = df['chembl_id'].tolist()
    smiles = df['smiles'].tolist()
    steric = 0.
    num_cliff_pairs = 0
    data = MoleculeDataset(dataset_name, DATA_PATH)   
    data()
    mol_cliff_dict = data.cliff_dict
    for key in mol_cliff_dict:
        if mol_cliff_dict[key][0]['is_cliff_mol']:
            idx0 = smiles.index(key)
            chembl_id0 = chembl_ids[idx0]
            v0 = calculate_molecular_volume(chembl_id0, key)
            if v0 == 0.:
                continue
            for mmp in mol_cliff_dict[key][1:]:
                mmp_smiles = mmp['smiles']
                idx1 = smiles.index(mmp_smiles)
                chembl_id1 = chembl_ids[idx1]
                v1 = calculate_molecular_volume(chembl_id1, mmp_smiles)
                if v1 == 0.:
                    continue
                steric += abs((v1 - v0)/v0)
                num_cliff_pairs += 1
    return steric/num_cliff_pairs

if __name__ == '__main__':
    steric_list = []
    for dataset_name in DATASETS[-9:]:
        if dataset_name in ['CHEMBL233_Ki']:
            steric_list.append(0.)
            continue

        steric = cal_steric_effect(dataset_name)
        print(f'{dataset_name}: {steric}')
        steric_list.append(steric)

    # write to a csv file
    import csv
    with open('./steric_effect.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'steric_effect'])
        for i in range(len(DATASETS)):
            writer.writerow([DATASETS[i], steric_list[i]])

