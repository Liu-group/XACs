import torch
from rf_utils import diff_mask
from featurization import MolTensorizer
from explain.GradCAM import GraphLayerGradCam
import numpy as np
from utils import pairwise_ranking_loss
from dataset import MoleculeDataset

def get_one_smiles_att(model, smiles) -> torch.Tensor:
    featurizer = MolTensorizer()
    data = featurizer.tensorize(smiles)
    model.eval()
    with torch.no_grad():
        explain_method = GraphLayerGradCam(model, model.convs[-1])
        att = explain_method.attribute((data.x, data.edge_attr), additional_forward_args=(data.edge_index))
        att = att.reshape(-1, 1)
    return att

def evaluate_gnn_explain_direction(data: MoleculeDataset, model):
    model.to('cpu')
    smiles_test = data.smiles_test
    cliff_dict = data.cliff_dict
    gnn_score = []
    loss = 0.0
    num_pairs = 0
    for smi in smiles_test:
        mmp_dicts = cliff_dict[smi]
        for mmp_dict in mmp_dicts[1:]:
            diff = mmp_dict['potency_diff']
            mmp_smi = mmp_dict['smiles']
            uncommon_atom_idx_i = mmp_dict['uncommon_atom_idx_i']
            uncommon_atom_idx_j = mmp_dict['uncommon_atom_idx_j']

            att_i = get_one_smiles_att(model, smi)
            att_j = get_one_smiles_att(model, mmp_smi)
            att_i_uncom = att_i[uncommon_atom_idx_i] if len(uncommon_atom_idx_i) > 0 else torch.zeros(1)
            att_j_uncom = att_j[uncommon_atom_idx_j] if len(uncommon_atom_idx_j) > 0 else torch.zeros(1)
            
            loss += pairwise_ranking_loss((torch.sum(att_i_uncom) - torch.sum(att_j_uncom)), torch.tensor(diff))
            num_pairs += 1
            score = 1 if (torch.sum(att_i_uncom) - torch.sum(att_j_uncom)) * diff > 0 else 0
            gnn_score.append(score)
    print("Total attribution loss: ", loss)
    print("Molecule averaged attribution loss: ", loss/len(smiles_test))
    print("Pair averaged attribution loss: ", loss/num_pairs)
    print("gnn direction score: ", np.mean(gnn_score))
    return np.mean(gnn_score)

def evaluate_rf_explain_direction(data, model_rf):
    smiles_test = data.smiles_test
    cliff_dict = data.cliff_dict
    rf_score = []
    for smi in smiles_test:
        mmp_dicts = cliff_dict[smi]
        for mmp_dict in mmp_dicts[1:]:
            diff = mmp_dict['potency_diff']
            mmp_smi = mmp_dict['smiles']
            uncommon_atom_idx_i = mmp_dict['uncommon_atom_idx_i']
            uncommon_atom_idx_j = mmp_dict['uncommon_atom_idx_j']
            smi_mask = diff_mask(smi, model_rf.predict)
            mmp_mask = diff_mask(mmp_smi, model_rf.predict)
            mask_pred_i_uncom = smi_mask[uncommon_atom_idx_i]
            mask_pred_j_uncom = mmp_mask[uncommon_atom_idx_j]
            score = 1 if (np.sum(mask_pred_i_uncom) - np.sum(mask_pred_j_uncom)) * diff > 0 else 0
            rf_score.append(score)
    print("rf direction score: ", np.mean(rf_score))
    return np.mean(rf_score)