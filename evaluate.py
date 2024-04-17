import collections  
import torch
from torch_geometric.data import Data
from rf_utils import diff_mask
from featurization import MolTensorizer
from explain.GradCAM import GraphLayerGradCam
from explain.InputXGrad import InputXGradient
import numpy as np
from utils import pairwise_ranking_loss
from dataset import MoleculeDataset
from GNN import GNN
from typing import List
from train import predict
from torch_geometric.loader import DataLoader
from metrics import get_metric_func

def get_gradcam_att(model: GNN, graph: Data) -> torch.Tensor:
    with torch.no_grad():
        explain_method = GraphLayerGradCam(model, model.convs[-1])
        node_weights = explain_method.attribute((graph.x, graph.edge_attr), additional_forward_args=(graph.edge_index))
    att = node_weights.cpu().reshape(-1, 1)
    return att

def get_inputxgrad_att(model: GNN, graph: Data) -> torch.Tensor:
    with torch.no_grad():
        explain_method = InputXGradient(model)
        node_weights, edge_weights = explain_method.attribute((graph.x, graph.edge_attr), additional_forward_args=(graph.edge_index))
        for idx in range(graph.num_edges):
            e_imp = edge_weights[idx]
            node_weights[graph.edge_index[0, idx]] += e_imp / 2
            node_weights[graph.edge_index[1, idx]] += e_imp / 2
    att = node_weights.cpu().reshape(-1, 1)
    return att

def get_graph_mask_att(model: GNN, graph: Data, masked_graphs: List[Data]) -> torch.Tensor:
    with torch.no_grad():
        pred = model(graph.x, graph.edge_attr, graph.edge_index)
        mod_preds = torch.tensor([model(masked_graph.x, masked_graph.edge_attr, masked_graph.edge_index) for masked_graph in masked_graphs])
    diff = pred - mod_preds
    diff = diff.cpu().reshape(-1, 1)
    return diff

def get_uncommon_att(att: torch.Tensor, uncommon_atom_idx: list) -> torch.Tensor:
    return att[uncommon_atom_idx] if len(uncommon_atom_idx) > 0 else torch.zeros(1)    
   
def evaluate_gnn_explain_direction(dataset: MoleculeDataset, data_test: Data, model: GNN):
    model.to('cpu')
    model.eval()
    smiles_test = [data_test[i].smiles for i in range(len(data_test))]
    cliff_dict = dataset.cliff_dict
    gnn_direction_score = collections.defaultdict(list)
    gradcam_loss, inputxgrad_loss = 0., 0.
    num_pairs = 0
    featurizer = MolTensorizer()
    for smi in smiles_test:
        print("smiles",smi)
        mmp_dicts = cliff_dict[smi]
        graph_i = featurizer.tensorize(smi)
        masked_graphs_i = featurizer.gen_masked_atom_feats(smi)
        gradcam_att_i = get_gradcam_att(model, graph_i)
        inputxgrad_att_i = get_inputxgrad_att(model, graph_i)
        diff_mask_i = get_graph_mask_att(model, graph_i, masked_graphs_i)
        for mmp_dict in mmp_dicts[1:]:
            num_pairs += 1
            diff = mmp_dict['potency_diff']
            mmp_smi = mmp_dict['smiles']
            uncommon_atom_idx_i = mmp_dict['uncommon_atom_idx_i']
            uncommon_atom_idx_j = mmp_dict['uncommon_atom_idx_j']
            
            graph_j = featurizer.tensorize(mmp_smi)
            masked_graphs_j = featurizer.gen_masked_atom_feats(mmp_smi)

            gradcam_att_j = get_gradcam_att(model, graph_j)
            inputxgrad_att_j = get_inputxgrad_att(model, graph_j)
            diff_mask_j = get_graph_mask_att(model, graph_j, masked_graphs_j)
            
            gnn_direction_score['gradcam'].append((diff, smi, mmp_smi, 1 if \
                (torch.sum(get_uncommon_att(gradcam_att_i, uncommon_atom_idx_i)) - \
                 torch.sum(get_uncommon_att(gradcam_att_j, uncommon_atom_idx_j))
                 ) * diff > 0 \
                 else 0))       
            gnn_direction_score['inputxgrad'].append((diff, smi, mmp_smi, 1 if \
                (torch.sum(get_uncommon_att(inputxgrad_att_i, uncommon_atom_idx_i)) - \
                 torch.sum(get_uncommon_att(inputxgrad_att_j, uncommon_atom_idx_j))
                 ) * diff > 0 \
                 else 0))

            gnn_direction_score['mask'].append((diff, smi, mmp_smi, 1 if \
                (torch.sum(get_uncommon_att(diff_mask_i, uncommon_atom_idx_i)) - \
                 torch.sum(get_uncommon_att(diff_mask_j, uncommon_atom_idx_j))
                 ) * diff > 0 \
                 else 0))

            gradcam_loss += pairwise_ranking_loss(
                    (torch.sum(get_uncommon_att(gradcam_att_i, uncommon_atom_idx_i)) - \
                     torch.sum(get_uncommon_att(gradcam_att_j, uncommon_atom_idx_j))
                     ),
                    torch.tensor(diff)).item()
            inputxgrad_loss += pairwise_ranking_loss(
                    (torch.sum(get_uncommon_att(inputxgrad_att_i, uncommon_atom_idx_i)) - \
                     torch.sum(get_uncommon_att(inputxgrad_att_j, uncommon_atom_idx_j))
                     ),
                    torch.tensor(diff)).item()
        if num_pairs>0:
            break
    print("Total number of ground_truth explanations", num_pairs)
    gradcam_score = np.mean([gnn_direction_score['gradcam'][i][3] for i in range(len(gnn_direction_score['gradcam']))])
    inputxgrad_score = np.mean([gnn_direction_score['inputxgrad'][i][3] for i in range(len(gnn_direction_score['inputxgrad']))])
    mask_score = np.mean([gnn_direction_score['mask'][i][3] for i in range(len(gnn_direction_score['mask']))])

    print("Total CradCAM attribution loss: {:.4f}".format(gradcam_loss))
    print("GradCAM Molecule averaged attribution loss: {:.4f}".format(gradcam_loss/len(smiles_test)))
    print("Pair averaged attribution loss: {:.4f}".format(gradcam_loss/num_pairs))
    print("Total InputXGrad attribution loss: {:.4f}".format(inputxgrad_loss))
    print("InputXGrad Molecule averaged attribution loss: {:.4f}".format(inputxgrad_loss/len(smiles_test)))
    print("Pair averaged attribution loss: {:.4f}".format(inputxgrad_loss/num_pairs))
    print("gnn GradCAM direction score: {:.4f}".format(gradcam_score))
    print("gnn InputXGrad direction score: {:.4f}".format(inputxgrad_score))
    print("gnn mask score: {:.4f}".format(mask_score))
    return {'gradcam': gradcam_score, 'inputxgrad': inputxgrad_score, 'mask': mask_score}, gnn_direction_score

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

def run_evaluation(args, model, data_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_loader = DataLoader(data_test, batch_size = args.batch_size, shuffle=False)
    loss_func = torch.nn.MSELoss()
    metric_func = get_metric_func(metric=args.metric)
    test_score, test_cliff_score, explan_acc = predict(args, model, test_loader, loss_func, metric_func, device)
    return test_score, test_cliff_score, explan_acc