import os
from typing import List, Union, Tuple
import numpy as np
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, rdFMCS
from Levenshtein import distance as levenshtein
from tqdm import tqdm
import collections
from rdkit.Chem.Scaffolds.MurckoScaffold import MakeScaffoldGeneric as GraphFramework
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from XACs.utils.const import TIMEOUT_MCS
from multiprocessing import Pool, cpu_count

NPROC = cpu_count()

class ActivityCliffs:
    """ Activity cliff class that find cliff pairs and
        computes uncom and com part of substructurein between based on MCS """
    def __init__(self, 
                 smiles: List[str], 
                 y_all: Union[List[float], np.array], 
                 struct_sim: Union[Tuple[str, float], str, None] = ('combined', 0.9),
                 dist_thre: float = 1.0,                 
                 dict_path: str = None,
                 ):
        """
        :param smiles: (list) list of SMILES strings
        :param y_all: (list) list of predicted target
        :param struct_sim: (metrics, float) structural similarity metric and threshold
        :param dist_thre: (float) the threshold of the target distance score
        :param dict_path: (str) the path to save the dictionary
        """
        self.smiles = smiles
        self.num_smiles = len(self.smiles)
        self.y_all = list(y_all) if type(y_all) is not list else y_all
        if os.path.exists(dict_path):
            if os.path.isfile(dict_path):
                with open(dict_path, 'rb') as f:
                    self.mcs_dict = pickle.load(f)
                    print(f"Loaded {dict_path}")
                    self.cliff_mols = self.get_cliff_mol_from_dict()
            else:
                raise ValueError(f"{dict_path} is not a file")

        else:
            print(f"{dict_path} does not exist, generating the mcs_dict...")
            self.struct_sim = struct_sim
            self.dist_thre = dist_thre
            if self.dist_thre == None:
                dist_mat = np.abs(np.array(y_all).reshape(-1, 1) - np.array(y_all).reshape(1, -1))
                self.dist_thre = np.mean(dist_mat[dist_mat > np.median(dist_mat)])
            if self.struct_sim == None:
                sim_mat = get_tanimoto_matrix(smiles)
                self.struct_sim = ('tanimoto', np.mean(sim_mat[sim_mat > np.median(sim_mat)]))
            if self.struct_sim == 'mmp':
                self.struct_sim = ('mmp',)

            self.dict_path = dict_path
            self.mcs_dict = self.find_cliffs()
            self.cliff_mols = self.get_cliff_mol_from_dict()
    
    def get_cliff_mol_from_dict(self):
        """
        Get the cliffs from the cliff dictionary.
        """
        cliff_mol = []
        for smiles in self.smiles:
            is_cliff_mol = self.mcs_dict[smiles][0]['is_cliff_mol']
            if is_cliff_mol:
                cliff_mol.append(1)
            else:
                cliff_mol.append(0)
        return cliff_mol
    
    def find_cliffs(self):
        """
        Find activity cliffs based on the similarity and potency fold change. If satisfied,
        get the matched molecular pair dictionary.  
        :return: (np.array) returns a binary matrix where 1 means activity cliff compounds
        """
        mcs_dict = collections.defaultdict(list)
        for smiles in self.smiles:
            mcs_dict[smiles].append({"is_cliff_mol": False})

        # if struct_sim is mmp, use mmpdb to generate mmp pairs
        if self.struct_sim[0] == 'mmp':
            print("Using mmpdb to find ACs...")
            smi_file = convert2smi(self.smiles, os.path.join(os.path.dirname(self.dict_path), 'smi_file.smi'))
            acs_df = mmp_is_cliff(calc_MMPs(smi_file), self.y_all, self.smiles, self.dist_thre)
            for i in range(len(acs_df)):
                smiles_i, smiles_j = acs_df.iloc[i]['smi_1'], acs_df.iloc[i]['smi_2']
                mol1, mol2 = Chem.MolFromSmiles(smiles_i), Chem.MolFromSmiles(smiles_j)
                num_atoms_i, num_atoms_j = mol1.GetNumAtoms(), mol2.GetNumAtoms()
                y_diff = acs_df.iloc[i]['y_diff']
                # common part
                constant_part = acs_df.iloc[i]['core']
                constant_sub = Chem.MolFromSmarts(constant_part)
                matched_atom_idx_i = list(mol1.GetSubstructMatch(constant_sub))
                matched_atom_idx_j = list(mol2.GetSubstructMatch(constant_sub))
                # uncommon part
                reactant, product = acs_df.iloc[i]['transformation'].split('>>')
                reactant_sub = Chem.MolFromSmarts(reactant)
                product_sub = Chem.MolFromSmarts(product)
                uncommon_atom_idx_i = list(mol1.GetSubstructMatch(reactant_sub))
                uncommon_atom_idx_j = list(mol2.GetSubstructMatch(product_sub))
                mmp_dict_i, mmp_dict_j = {}, {}
                mmp_dict_i['smiles'], mmp_dict_j['smiles'] = smiles_j, smiles_i
                mmp_dict_i['uncommon_atom_idx_i'], mmp_dict_j['uncommon_atom_idx_i'] = uncommon_atom_idx_i, uncommon_atom_idx_j
                mmp_dict_i['uncommon_atom_idx_j'], mmp_dict_j['uncommon_atom_idx_j'] = uncommon_atom_idx_j, uncommon_atom_idx_i
                mmp_dict_i['common_atom_idx_i'], mmp_dict_j['common_atom_idx_i'] = matched_atom_idx_i, matched_atom_idx_j
                mmp_dict_i['common_atom_idx_j'], mmp_dict_j['common_atom_idx_j'] = matched_atom_idx_j, matched_atom_idx_i
                mmp_dict_i['atom_mask_i'], mmp_dict_j['atom_mask_i'] = [1 if i in uncommon_atom_idx_i else 0 for i in range(num_atoms_i)], [1 if i in uncommon_atom_idx_j else 0 for i in range(num_atoms_j)]
                mmp_dict_i['atom_mask_j'], mmp_dict_j['atom_mask_j'] = [1 if i in uncommon_atom_idx_j else 0 for i in range(num_atoms_j)], [1 if i in uncommon_atom_idx_i else 0 for i in range(num_atoms_i)]
                mcs_dict[smiles_i].append(mmp_dict_i)
                mcs_dict[smiles_j].append(mmp_dict_j)

                mcs_dict[smiles_i][0]['is_cliff_mol'] = True
                mcs_dict[smiles_j][0]['is_cliff_mol'] = True
                mmp_dict_i['potency_diff'] = y_diff if type(y_diff) is float else min(y_diff, 1)
                mmp_dict_j['potency_diff'] = -y_diff if type(y_diff) is float else min(-y_diff, 1)
                mcs_dict[smiles_i].append(mmp_dict_i)
                mcs_dict[smiles_j].append(mmp_dict_j)
            with open(self.dict_path, 'wb') as f:
                pickle.dump(mcs_dict, f)
            print(f'{self.dict_path} saved')
            return mcs_dict

        pool = Pool(NPROC)    
        print(f"Finding activity cliffs using {NPROC} cpus...")
        results = []
        for i in tqdm(range(self.num_smiles)):
            smiles_i = self.smiles[i]
            y_i = self.y_all[i]
            for j in range(i + 1, self.num_smiles):
                smiles_j = self.smiles[j]
                y_j = self.y_all[j]
                result = pool.apply_async(if_cliff, args=(smiles_i, 
                                                          smiles_j, 
                                                          y_i, 
                                                          y_j, 
                                                          self.struct_sim, 
                                                          self.dist_thre))
                results.append([i, j, result])
        asyncresults = [[res[0], res[1]] + [res[2].get()] for res in results]
        for asyncresult in asyncresults:    
            i, j, (mmp_dict_i, mmp_dict_j) = asyncresult
            smiles_i, smiles_j = self.smiles[i], self.smiles[j]
            y_i, y_j = self.y_all[i], self.y_all[j]#-np.log10(self.y[i]), -np.log10(self.y[j])
            if mmp_dict_i is not None:
                mcs_dict[smiles_i][0]['is_cliff_mol'] = True
                mcs_dict[smiles_j][0]['is_cliff_mol'] = True
                if mmp_dict_i is not True:
                    mmp_dict_i['potency_diff'] = (y_i - y_j) if type(y_i) is float else min((y_j - y_i), 1)
                    mmp_dict_j['potency_diff'] = (y_j - y_i) if type(y_j) is float else min((y_i - y_j), 1)
                    mcs_dict[smiles_i].append(mmp_dict_i)
                    mcs_dict[smiles_j].append(mmp_dict_j)
            else:   
                continue
        pool.close()
        # save mcs_dict as pkl file
        with open(self.dict_path, 'wb') as f:
            pickle.dump(mcs_dict, f)
        print(f'{self.dict_path} saved')
        return mcs_dict

def if_cliff(smiles_i, smiles_j, y_i, y_j, struct_sim: Tuple[str, float], dist_threshold: float = 1.0):
    """
    Judge whether the pair of molecules is a cliff based on the similarity and potency fold change.
    If satisfied, get the matched molecular pair dictionary based on the maximum common substructure.
    """    
    diff = abs(y_i - y_j)
    if diff >= dist_threshold:
        mmp = moleculeace_similarity(smiles_i, smiles_j, struct_sim)
        if mmp:
            mmp_dict_i, mmp_dict_j = get_mcs(smiles_i, smiles_j)
            if mmp_dict_i is None:
                return True, True
            return mmp_dict_i, mmp_dict_j
    return None, None

def get_mcs(smiles_i, smiles_j, mcs_defualt: bool = False):
    """Get the maximum common substructure of two molecules and return as a dictionary."""
    mol_i, mol_j = Chem.MolFromSmiles(smiles_i), Chem.MolFromSmiles(smiles_j)
    num_atoms_i, num_atoms_j = mol_i.GetNumAtoms(), mol_j.GetNumAtoms()
    atom_indices_i, atom_indices_j = list(range(num_atoms_i)), list(range(num_atoms_j))

    if mcs_defualt:
        mcs = rdFMCS.FindMCS([mol_i, mol_j], timeout=TIMEOUT_MCS)
    else:
        mcs = rdFMCS.FindMCS([mol_i, mol_j],
                            matchValences=True,
                            ringMatchesRingOnly=True,
                            completeRingsOnly=True,
                            timeout=TIMEOUT_MCS)
    if not mcs.canceled:        
        substru_smi = mcs.smartsString
        substru_mol = Chem.MolFromSmarts(substru_smi)
        substru_atoms_mol_i = mol_i.GetSubstructMatch(substru_mol)
        substru_atoms_mol_j = mol_j.GetSubstructMatch(substru_mol)    
        matched_atom_idx_i = [atom_idx for atom_idx in substru_atoms_mol_i]
        matched_atom_idx_j = [atom_idx for atom_idx in substru_atoms_mol_j]
    else:
        if mcs.canceled:
            print(f"Timeout for {smiles_i} and {smiles_j}")
        return None, None

    mmp_dict_i, mmp_dict_j = {}, {}
    uncommon_atom_idx_i = [atom_idx for atom_idx in atom_indices_i if atom_idx not in matched_atom_idx_i]
    uncommon_atom_idx_j = [atom_idx for atom_idx in atom_indices_j if atom_idx not in matched_atom_idx_j]
    mmp_dict_i['smiles'], mmp_dict_j['smiles'] = smiles_j, smiles_i
    mmp_dict_i['uncommon_atom_idx_i'], mmp_dict_j['uncommon_atom_idx_i'] = uncommon_atom_idx_i, uncommon_atom_idx_j
    mmp_dict_i['uncommon_atom_idx_j'], mmp_dict_j['uncommon_atom_idx_j'] = uncommon_atom_idx_j, uncommon_atom_idx_i
    mmp_dict_i['common_atom_idx_i'], mmp_dict_j['common_atom_idx_i'] = matched_atom_idx_i, matched_atom_idx_j
    mmp_dict_i['common_atom_idx_j'], mmp_dict_j['common_atom_idx_j'] = matched_atom_idx_j, matched_atom_idx_i
    mmp_dict_i['atom_mask_i'], mmp_dict_j['atom_mask_i'] = [1 if i in uncommon_atom_idx_i else 0 for i in range(num_atoms_i)], [1 if i in uncommon_atom_idx_j else 0 for i in range(num_atoms_j)]
    mmp_dict_i['atom_mask_j'], mmp_dict_j['atom_mask_j'] = [1 if i in uncommon_atom_idx_j else 0 for i in range(num_atoms_j)], [1 if i in uncommon_atom_idx_i else 0 for i in range(num_atoms_i)]
    return mmp_dict_i, mmp_dict_j

def get_tanimoto_matrix(smiles: List[str], radius: int = 2, nBits: int = 1024):
    """ Calculates a matrix of Tanimoto similarity scores for a list of SMILES string"""
    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    # Calculate upper triangle of matrix
    print("Getting tanimoto matrix...")
    for i in tqdm(range(smi_len)):
        for j in range(i+1, smi_len):
            m[i, j] = get_tanimoto_score(smiles[i], smiles[j], radius=radius, nBits=nBits)
    # Fill in the lower triangle without having to loop (saves ~50% of time)
    m = m + m.T - np.diag(np.diag(m))
    # Fill the diagonal with 0's
    np.fill_diagonal(m, 0)

    return m

def get_scaffold_score(smiles_i: str, smiles_j: str, radius: int = 2, nBits: int = 1024):
    mol_i, mol_j = Chem.MolFromSmiles(smiles_i), Chem.MolFromSmiles(smiles_j)
    try:
        skeleton_i, skeleton_j = GraphFramework(mol_i), GraphFramework(mol_j)
    except Exception:  # In the very rare case this doesn't work, use a normal scaffold
        print(f"Could not create a generic scaffold of {smiles_i or smiles_j}, used a normal scaffold instead")
        skeleton_i, skeleton_j = GetScaffoldForMol(mol_i), GetScaffoldForMol(mol_j)
    skeleton_fp_i = AllChem.GetMorganFingerprintAsBitVect(skeleton_i, radius=radius, nBits=nBits)
    skeleton_fp_j = AllChem.GetMorganFingerprintAsBitVect(skeleton_j, radius=radius, nBits=nBits)
    score = DataStructs.TanimotoSimilarity(skeleton_fp_i, skeleton_fp_j)
    return score

def get_tanimoto_score(smiles_i: str, smiles_j: str, radius: int = 2, nBits: int = 1024):
    mol_i, mol_j = Chem.MolFromSmiles(smiles_i), Chem.MolFromSmiles(smiles_j)
    fp_i  = AllChem.GetMorganFingerprintAsBitVect(mol_i, radius=radius, nBits=nBits)
    fp_j = AllChem.GetMorganFingerprintAsBitVect(mol_j, radius=radius, nBits=nBits)
    score = DataStructs.TanimotoSimilarity(fp_i, fp_j)
    return score

def get_levenshtein_score(smiles_i: str, smiles_j: str, normalize: bool = True):
    """ Calculates the levenshtein similarity scores for a two of SMILES string"""
    if normalize:
        score = 1 - levenshtein(smiles_i, smiles_j) / max(len(smiles_i), len(smiles_j))
    else:
        score = 1 - levenshtein(smiles_i, smiles_j)
    # Get from a distance to a similarity
    return score

def moleculeace_similarity(smiles_i: str, smiles_j: str, struct_sim: Tuple[str, float]):
    """ Calculate whether the pairs of molecules have a high tanimoto, scaffold, or SMILES similarity """
    threshold = struct_sim[1]
    if struct_sim[0] == 'tanimoto':
        score = get_tanimoto_score(smiles_i, smiles_j) >= threshold
        return score
    else:
        score_tani = get_tanimoto_score(smiles_i, smiles_j) >= threshold
        score_scaff = get_scaffold_score(smiles_i, smiles_j) >= threshold
        score_leve = get_levenshtein_score(smiles_i, smiles_j) >= threshold
        return any([score_tani, score_scaff, score_leve])

def convert2smi(smiles_list: List[str], smi_file: str = None):
    """
    Convert data to smi file for the use of mmpdb.
    """
    smiles_with_id = pd.DataFrame({
            'smiles': smiles_list,
            'id': range(1, len(smiles_list) + 1)  # Adding a simple sequential identifier
            })
    smiles_with_id.to_csv(smi_file, sep=' ', index=False, header=False)
    print(f"SMILES data saved to {smi_file}")
    return smi_file

def calc_MMPs(smifile):
    """
    Generate MMP Indexing and Matching using mmpdb
    """
    # TODO switch system calls to just importing the python code and using it directly
    print("Generating MMP Fragments")
    fragfile = os.path.join(os.path.dirname(smifile), 'frag_file.frag')
    mmpdb_out = os.path.join(os.path.dirname(smifile), 'mmps.csv')
    os.system(f'mmpdb fragment {smifile} --num-jobs {NPROC} --num-cuts 1 -o {fragfile}')
    os.system(f"mmpdb index {fragfile} -s\
                                     --max-variable-ratio 0.33 \
                                     --max-heavies-transf 8 \
                                     -o {mmpdb_out} \
                                     --max-variable-heavies 13 \
                                     --out 'csv'")
    print(f"MMPs saved to {mmpdb_out}")
    mmpdb_out_df = pd.read_csv(mmpdb_out, sep='\t', names=['smi_1', 'smi_2', 'idx1', 'idx2', 'transformation', 'core'])
    return mmpdb_out_df

def mmp_is_cliff(mmpdb_out_df, y_all, smiles, dist_thre):
    """
    Get ACs based on matched molecular pair.
    """
    data = pd.DataFrame({'smiles': smiles, 'y': y_all})
    merged1 = mmpdb_out_df.merge(data.rename(columns={'smiles': 'smi_1', 'y': 'y1'}), on='smi_1')
    merged2 = merged1.merge(data.rename(columns={'smiles': 'smi_2', 'y': 'y2'}), on='smi_2')
    # get the absolute difference of the y1 and y2
    merged2['y_diff'] = merged2['y1'] - merged2['y2']
    merged2 = merged2[abs(merged2['y_diff']) >= dist_thre]
    return merged2