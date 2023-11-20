import torch
from rf_utils import diff_mask
from featurization import MolTensorizer
from explain.GradCAM import GraphLayerGradCam
from explain.InputXGrad import InputXGradient
import numpy as np
from utils import pairwise_ranking_loss
from dataset import MoleculeDataset


def get_one_smiles_att(model, smiles) -> torch.Tensor:
    featurizer = MolTensorizer()
    data = featurizer.tensorize(smiles)
    model.eval()
    ''' 
       if args.att_method == 'GradCAM':
            explain_method = GraphLayerGradCam(model, model.convs[-1])
            node_weights = explain_method.attribute((data.x, data.edge_attr), additional_forward_args=(data.edge_index))
        if args.att_method == 'InputXGrad':
            explain_method = InputXGradient(model)
            node_weights, edge_weights = explain_method.attribute((data.x, data.edge_attr), additional_forward_args=(data.edge_index))
            for idx in range(data.num_edges):
                e_imp = edge_weights[idx]
                node_weights[data.edge_index[0, idx]] += e_imp / 2
                node_weights[data.edge_index[1, idx]] += e_imp / 2
    '''
    with torch.no_grad():
        explain_method = GraphLayerGradCam(model, model.convs[-1])
        node_weights = explain_method.attribute((data.x, data.edge_attr), additional_forward_args=(data.edge_index))
        att = node_weights.cpu().reshape(-1, 1)
        masked_graphs = featurizer.gen_masked_atom_feats(smiles)
        pred = model(data.x, data.edge_attr, data.edge_index)
        mod_preds = torch.tensor([model(masked_graph.x, masked_graph.edge_attr, masked_graph.edge_index) for masked_graph in masked_graphs])
        diff = pred - mod_preds
        diff = diff.cpu().reshape(-1, 1)

    return att, diff

def evaluate_gnn_explain_direction(data: MoleculeDataset, model):
    model.to('cpu')
    smiles_test = [data.data_test[i].smiles for i in range(len(data.data_test))]
    cliff_dict = data.cliff_dict
    gnn_score = []
    gnn_mask_score = []
    loss = 0.0
    num_pairs = 0
    for smi in smiles_test:
        mmp_dicts = cliff_dict[smi]
        att_i, diff_i = get_one_smiles_att(model, smi)
        for mmp_dict in mmp_dicts[1:]:
            diff = mmp_dict['potency_diff']
            mmp_smi = mmp_dict['smiles']
            uncommon_atom_idx_i = mmp_dict['uncommon_atom_idx_i']
            uncommon_atom_idx_j = mmp_dict['uncommon_atom_idx_j']

            att_j, diff_j = get_one_smiles_att(model, mmp_smi)

            att_i_uncom = att_i[uncommon_atom_idx_i] if len(uncommon_atom_idx_i) > 0 else torch.zeros(1)
            att_j_uncom = att_j[uncommon_atom_idx_j] if len(uncommon_atom_idx_j) > 0 else torch.zeros(1)
            
            loss += pairwise_ranking_loss((torch.sum(att_i_uncom) - torch.sum(att_j_uncom)), torch.tensor(diff))
            num_pairs += 1
            score = 1 if (torch.sum(att_i_uncom) - torch.sum(att_j_uncom)) * diff > 0 else 0

            diff_i_uncom = diff_i[uncommon_atom_idx_i] if len(uncommon_atom_idx_i) > 0 else torch.zeros(1)
            diff_j_uncom = diff_j[uncommon_atom_idx_j] if len(uncommon_atom_idx_j) > 0 else torch.zeros(1)
            
            mask_score = 1 if (torch.sum(diff_i_uncom) - torch.sum(diff_j_uncom)) * diff > 0 else 0
            gnn_score.append(score)
            gnn_mask_score.append(mask_score)
    print("Total attribution loss: ", loss)
    print("Molecule averaged attribution loss: ", loss/len(smiles_test))
    print("Pair averaged attribution loss: ", loss/num_pairs)
    print("gnn direction score: ", np.mean(gnn_score))
    print("gnn mask score: ", np.mean(gnn_mask_score))
    return np.mean(gnn_score)

def evaluate_rf_explain_direction(data, model_rf):
    smiles_test = [data.data_test[i].smiles for i in range(len(data.data_test))]
    cliff_dict = data.cliff_dict
    rf_score = []
    for smi in smiles_test:
        mmp_dicts = cliff_dict[smi]
        smi_mask = diff_mask(smi, model_rf.predict)
        for mmp_dict in mmp_dicts[1:]:
            diff = mmp_dict['potency_diff']
            mmp_smi = mmp_dict['smiles']
            uncommon_atom_idx_i = mmp_dict['uncommon_atom_idx_i']
            uncommon_atom_idx_j = mmp_dict['uncommon_atom_idx_j']
            
            mmp_mask = diff_mask(mmp_smi, model_rf.predict)
            mask_pred_i_uncom = smi_mask[uncommon_atom_idx_i]
            mask_pred_j_uncom = mmp_mask[uncommon_atom_idx_j]
            score = 1 if (np.sum(mask_pred_i_uncom) - np.sum(mask_pred_j_uncom)) * diff > 0 else 0
            rf_score.append(score)
    print("Total number of ground_truth explanations", len(rf_score))
    print("rf direction score: ", np.mean(rf_score))
    return np.mean(rf_score)