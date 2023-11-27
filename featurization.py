import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Mol
from rdkit.Chem.rdchem import Atom, Bond, Mol
from torch_geometric.data import Data
from typing import List, Union
import numpy as np
from const import ATOM_TYPES, BOND_TYPES, STEREO_TYPES
import numpy as np
from rf_utils import gen_dummy_atoms

def one_hot_encoding(x, allowable_set):
    """One-hot encoding.
    Parameters
    ----------
    x : str, int or Chem.rdchem.HybridizationType
    allowable_set : list
        The elements of the allowable_set should be of the
        same type as x.
    Returns
    -------
    list
        List of int (0 or 1) where at most one value is 1.
        If the i-th value is 1, then we must have x == allowable_set[i].
    """
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(int, list(map(lambda s: x == s, allowable_set))))


def multi_hot_encoding(x, allowable_set):
    """Multi-hot encoding.
    Args:
        x (list): any type that can be compared with elements in allowable_set
        allowable_set (list): allowed values for x to take
    Returns:
        list: List of int (0 or 1) where zero or more values can be 1.
            If the i-th value is 1, then we must have allowable_set[i] in x.
    """
    return list(map(int, list(map(lambda s: s in x, allowable_set))))

def get_pos(mol: Mol) -> torch.Tensor:
    AllChem.EmbedMolecule(mol,randomSeed=0xf00d)
    N = mol.GetNumAtoms()
    pos = Chem.MolToMolBlock(mol).split('\n')[4:4 + N]
    pos = [[float(x) for x in line.split()[:3]] for line in pos]
    return torch.tensor(pos)

class MolTensorizer(object):
    def __init__(self, 
                 featurization='normal',
                 ):
        self.featurization = featurization
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        self.num_node_feature = len(self.atom_features(unrelated_mol.GetAtomWithIdx(0)))
        self.num_bond_feature = len(self.bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

    def atom_features(
        self, atom: Atom, use_chirality: bool = False, hydrogens_implicit: bool = False
    ) -> List[float]:
        """
        Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
        """
        # define list of permitted atoms
        atom_types = ATOM_TYPES
        if hydrogens_implicit == True:
            atom_types = ["H"] + atom_types
        # compute atom features
        atom_type_enc = one_hot_encoding(str(atom.GetSymbol()), atom_types)
        implicit_valence_enc = one_hot_encoding(
            int(atom.GetImplicitValence()), [0, 1, 2, 3, 4, "MoreThanFour"]
        )
        n_heavy_neighbors_enc = one_hot_encoding(
            int(atom.GetDegree()), [1, 2, 3, 4, "MoreThanFour"]
        )
        formal_charge_enc = one_hot_encoding(
            int(atom.GetFormalCharge()), [-3, -2, -1, 0, 1, 2, 3, "Extreme"]
        )
        hybridisation_type_enc = one_hot_encoding(
            str(atom.GetHybridization()),
            ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"],
        )

        is_in_a_ring_enc = [int(atom.IsInRing())]

        is_aromatic_enc = [int(atom.GetIsAromatic())]

        atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]

        vdw_radius_scaled = [float((Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)]

        covalent_radius_scaled = [float((Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) / 0.76)]

        atom_feature_vector = (
            atom_type_enc
            + implicit_valence_enc
            + n_heavy_neighbors_enc
            + formal_charge_enc
            + hybridisation_type_enc
            + is_in_a_ring_enc
            + is_aromatic_enc
            + atomic_mass_scaled
            + vdw_radius_scaled
            + covalent_radius_scaled
        )

        if use_chirality:
            chirality_type_enc = one_hot_encoding(
                str(atom.GetChiralTag()),
                [
                    "CHI_UNSPECIFIED",
                    "CHI_TETRAHEDRAL_CW",
                    "CHI_TETRAHEDRAL_CCW",
                    "CHI_OTHER",
                ],
            )
            atom_feature_vector += chirality_type_enc
        if hydrogens_implicit == True:
            n_hydrogens_enc = one_hot_encoding(
                int(atom.GetTotalNumHs()), [0, 1, 2, 3, 4, "MoreThanFour"]
            )
            atom_feature_vector += n_hydrogens_enc
        return np.array(atom_feature_vector)
        
    def bond_features(self, bond: Bond, use_stereochemistry: bool = False) -> np.ndarray:
        """
        Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
        """

        bond_type_enc = one_hot_encoding(str(bond.GetBondType()), BOND_TYPES)

        bond_is_conj_enc = [int(bond.GetIsConjugated())]

        bond_is_in_ring_enc = [float(int(bond.IsInRing()))]

        bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

        if use_stereochemistry == True:
            stereo_type_enc = one_hot_encoding(
                str(bond.GetStereo()), STEREO_TYPES
            )
            bond_feature_vector += stereo_type_enc

        return np.array(bond_feature_vector)
    
    def tensorize(self, smile: Union[str, Mol]) -> Data:
        if isinstance(smile, str):
            mol = Chem.MolFromSmiles(smile)
        else:
            mol = smile
        # Get atom features
        xs = []
        for atom in mol.GetAtoms():
            xs.append(self.atom_features(atom))
        x = np.array(xs)   
        x = torch.tensor(x).view(-1, self.num_node_feature)

        # Get bond features
        edge_indices, edge_attrs = [], []
        # If no bonds (e.g. H2S), create an artifact bond
        if mol.GetNumBonds() == 0:
            e = [0] * self.num_bond_feature
            edge_indices = [[0, 0], [0, 0]]
            edge_attrs = [e, e]
        else:
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                e = self.bond_features(bond)
                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().type(torch.LongTensor).view(2, -1)
        edge_attrs = np.array(edge_attrs)
        edge_attr = torch.tensor(edge_attrs).view(-1, self.num_bond_feature)

        # Sort indices.
        if edge_index.numel() > 0:
            perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def gen_masked_atom_feats(self, smiles: str) -> List[Data]: 
        """
        Given a smiles, returns a list of graphs data where individual atoms
        are masked.
        """
        mol = Chem.MolFromSmiles(smiles)
        masked_mols = gen_dummy_atoms(mol)
        masked_graphs = []
        for masked_mol in masked_mols:
            masked_graphs.append(self.tensorize(masked_mol))
        return masked_graphs
