############################
### Modified from molucn####
############################
from copy import deepcopy
from typing import Callable, List, Tuple
import numpy as np
from rdkit.Chem import AllChem, DataStructs, MolFromSmiles, rdchem


FP_SIZE = 1024
BOND_RADIUS = 2


def gen_dummy_atoms(mol: rdchem.Mol, dummy_atom_no: int = 55) -> List[rdchem.Mol]:
    """
    Given a specific rdkit mol, returns a list of mols where each individual atom
    has been replaced by a dummy atom type.
    """
    mod_mols = []
    for idx_atom in range(mol.GetNumAtoms()):
        mol_cpy = deepcopy(mol)
        mol_cpy.GetAtomWithIdx(idx_atom).SetAtomicNum(dummy_atom_no)
        mod_mols.append(mol_cpy)
    return mod_mols


def featurize_ecfp4(mol: rdchem.Mol, fp_size=FP_SIZE, bond_radius=BOND_RADIUS):
    """
    Gets an ECFP4 fingerprint for a specific rdkit mol.
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, bond_radius, nBits=fp_size)
    arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def diff_mask(
    mol_string: str,
    pred_fun: Callable,
    fp_size: int = 1024,
    bond_radius: int = 2,
    dummy_atom_no: int = 47,
    mol_read_f: Callable = MolFromSmiles,
):
    """
    Given a mol specified by a string (SMILES, inchi), uses Sheridan's method (2019)
    alongside an sklearn model to compute atom attribution.
    """
    mol = mol_read_f(mol_string)
    og_fp = featurize_ecfp4(mol, fp_size, bond_radius)
    og_pred = pred_fun(og_fp[np.newaxis, :]).squeeze()

    mod_mols = gen_dummy_atoms(mol, dummy_atom_no)
    mod_fps = [featurize_ecfp4(mol, fp_size, bond_radius) for mol in mod_mols]

    mod_fps = np.vstack(mod_fps)
    mod_preds = pred_fun(mod_fps).squeeze()

    return og_pred - mod_preds
