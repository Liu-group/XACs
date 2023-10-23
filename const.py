from rdkit import Chem


TIMEOUT_MCS = 1800
ATOM_TYPES = [
        "B", 
        "C",
        "N",
        "O",
        "S",
        "F",
        "P",
        "Cl",
        "Br",
        "I",
        "*",
    ]
BOND_TYPES = ['SINGLE','DOUBLE','TRIPLE','AROMATIC']
STEREO_TYPES = ["STEREOZ", "STEREOE", "STEREONONE"]