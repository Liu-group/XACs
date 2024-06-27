# add {"is_cliff_mol": False} to each exstisted mcs_dict
from dataset import DATASETS, WORKING_DIR
import pickle
import os
from dataset import MoleculeDataset
from cliffs import moleculeace_similarity, find_fc
import numpy as np
for dataset in DATASETS[1:]:
    print(dataset)
    data = MoleculeDataset(dataset) 
    smiles = data.smiles_all
    bioactivity_orig = data.y_all
    bioactivity = 10 ** abs(np.array(bioactivity_orig))
    dict_path = os.path.join(WORKING_DIR, dataset, f'mcs_dict_0.9.pkl')
    if os.path.exists(os.path.join(WORKING_DIR, dataset, f'mcs_dict_0.9_copy.pkl')):
        continue
    if os.path.exists(dict_path):
        with open(dict_path, 'rb') as f:
            mcs_dict = pickle.load(f)

            for i in range(len(smiles)):
                smi_i = smiles[i]
                bioactivity_i = bioactivity[i]
                for j in range(i+1, len(smiles)):
                    smi_j = smiles[j]
                    bioactivity_j = bioactivity[j]
                    diff = find_fc(bioactivity_i, bioactivity_j)
                    mmp = moleculeace_similarity(smi_i, smi_j, similarity=0.9)
                    if diff > 10.0 and mmp:
                        if smi_i not in mcs_dict or (mcs_dict[smi_i][0] != {"is_cliff_mol": False} and mcs_dict[smi_i][0] != {"is_cliff_mol": True}):
                            mcs_dict[smi_i].insert(0, {"is_cliff_mol": True})
                        else:
                            mcs_dict[smi_i][0] = {"is_cliff_mol": True}
                        if smi_j not in mcs_dict or (mcs_dict[smi_j][0] != {"is_cliff_mol": False} and mcs_dict[smi_j][0] != {"is_cliff_mol": True}):
                            mcs_dict[smi_j].insert(0, {"is_cliff_mol": True})
                        else:
                            mcs_dict[smi_j][0] = {"is_cliff_mol": True}
                    elif smi_i not in mcs_dict or (mcs_dict[smi_i][0] != {"is_cliff_mol": True} and mcs_dict[smi_i][0] != {"is_cliff_mol": False}):
                        mcs_dict[smi_i].insert(0, {"is_cliff_mol": False})
                        if smi_j not in mcs_dict or (mcs_dict[smi_j][0] != {"is_cliff_mol": True} and mcs_dict[smi_j][0] != {"is_cliff_mol": False}):
                            mcs_dict[smi_j].insert(0, {"is_cliff_mol": False})   
                        else:
                            continue
                    else:
                        if smi_j not in mcs_dict or (mcs_dict[smi_j][0] != {"is_cliff_mol": True} and mcs_dict[smi_j][0] != {"is_cliff_mol": False}):
                            mcs_dict[smi_j].insert(0, {"is_cliff_mol": False})
                        else:
                            continue

        with open(os.path.join(WORKING_DIR, dataset, f'mcs_dict_0.9_copy.pkl'), 'wb') as f: 
            pickle.dump(mcs_dict, f)
        print('_copy.pkl saved')
    else:
        print(f'{dataset} mcs_dict_0.9.pkl does not exist')
        continue