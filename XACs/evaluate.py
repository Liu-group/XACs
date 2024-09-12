import collections  
from typing import List
import numpy as np
from XACs.utils.metrics import get_metric_func
from XACs.utils.rf_utils import diff_mask
from XACs.utils.utils import pairwise_ranking_loss
from XACs.attribution import GradCAM, InputXGrad, IG, SmoothGrad
from XACs.dataset import MoleculeDataset
from XACs.featurization import MolTensorizer
from XACs.GNN import GNN
from XACs.train import predict
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score

def get_gradcam_att(model: GNN, graph: Data) -> torch.Tensor:
    with torch.no_grad():
        explain_method = GradCAM.GraphLayerGradCam(model, model.convs[-1])
        node_weights = explain_method.attribute((graph.x, graph.edge_attr), additional_forward_args=(graph.edge_index))
    att = node_weights.cpu().reshape(-1, 1)
    return att

def get_inputxgrad_att(model: GNN, graph: Data, include_edge: bool = True) -> torch.Tensor:
    with torch.no_grad():
        explain_method = InputXGrad.InputXGradient(model)
        node_weights, edge_weights = explain_method.attribute((graph.x, graph.edge_attr), additional_forward_args=(graph.edge_index))
        if include_edge:
            for idx in range(graph.num_edges):
                e_imp = edge_weights[idx]
                node_weights[graph.edge_index[0, idx]] += e_imp / 2
                node_weights[graph.edge_index[1, idx]] += e_imp / 2
    att = node_weights.cpu().reshape(-1, 1)
    return att

def get_smoothgrad_att(model: GNN, graph: Data, noise_var: float = 0.15, num_samples: int = 50, include_edge: bool = True) -> torch.Tensor:
    with torch.no_grad():
        explain_method = SmoothGrad.SmoothGrad(model, noise_var, num_samples)
        node_weights, edge_weights = explain_method.attribute((graph.x, graph.edge_attr), additional_forward_args=(graph.edge_index))
        if include_edge:
            for idx in range(graph.num_edges):
                e_imp = edge_weights[idx]
                node_weights[graph.edge_index[0, idx]] += e_imp / 2
                node_weights[graph.edge_index[1, idx]] += e_imp / 2
    att = node_weights.cpu().reshape(-1, 1)
    return att

def get_graph_mask_att(model: GNN, graph: Data, masked_graph: List[Data]) -> torch.Tensor:
    with torch.no_grad():
        pred = model(graph.x, graph.edge_attr, graph.edge_index)
        mod_preds = model(masked_graph.x, masked_graph.edge_attr, masked_graph.edge_index)
    diff = pred - mod_preds
    diff = diff.cpu().reshape(-1, 1)
    return diff

def get_ig_att(model: GNN, graph: Data, include_edge: bool = True) -> torch.Tensor:
    with torch.no_grad():
        explain_method = IG.IntegratedGradient(model)
        node_weights, edge_weights = explain_method.attribute(graph.x, graph.edge_attr, baselines=None, edge_index = graph.edge_index)
        node_weights = node_weights.sum(dim=1)
        edge_weights = edge_weights.sum(dim=1)
        if include_edge:
            for idx in range(graph.num_edges):
                e_imp = edge_weights[idx]
                node_weights[graph.edge_index[0, idx]] += e_imp / 2
                node_weights[graph.edge_index[1, idx]] += e_imp / 2
    att = node_weights.cpu().reshape(-1, 1)
    return att

def get_attention_score(model: GNN, graph: Data) -> torch.Tensor:
    node_weights = torch.zeros(graph.num_nodes, 1)
    with torch.no_grad():
        pred, attention_weights = model(graph.x, graph.edge_attr, graph.edge_index, return_attention_weights=True)
        for adjs, att in attention_weights:
            for idx in range(adjs.shape[1]):
                e_imp = att[idx].mean()
                node_weights[adjs[0, idx]] += e_imp / 2
                node_weights[adjs[1, idx]] += e_imp / 2
    return att


def get_uncommon_att(att: torch.Tensor, uncommon_atom_idx: list) -> torch.Tensor:
    return att[uncommon_atom_idx] if len(uncommon_atom_idx) > 0 else torch.zeros(1)    
   
def evaluate_gnn_explain_direction(dataset: MoleculeDataset, data_test: Data, model: GNN):
    model.to('cpu')
    model.eval()
    smiles_test = [data_test[i].smiles for i in range(len(data_test))]
    cliff_dict = dataset.cliff_dict
    gnn_direction_score = collections.defaultdict(list)
    num_pairs = 0
    ground_truths = []
    featurizer = MolTensorizer()
    print("Evaluating feature attributions...")
    for smi in smiles_test:
        mmp_dicts = cliff_dict[smi]
        if mmp_dicts[0]['is_cliff_mol']==False:
            continue
        graph_i = featurizer.tensorize(smi)
        #masked_graphs_i = featurizer.gen_masked_atom_feats(smi)

        if model.conv_name == 'gat':
            attention_i = get_attention_score(model, graph_i)
        smoothgrad_att_i = get_smoothgrad_att(model, graph_i)
        gradcam_att_i = get_gradcam_att(model, graph_i)
        inputxgrad_att_i = get_inputxgrad_att(model, graph_i)
        #diff_mask_i = get_graph_mask_att(model, graph_i, masked_graphs_i)
        ig_att_i = get_ig_att(model, graph_i)
        for mmp_dict in mmp_dicts[1:]:
            num_pairs += 1
            diff = mmp_dict['potency_diff']
            mmp_smi = mmp_dict['smiles']
            uncommon_atom_idx_i = mmp_dict['uncommon_atom_idx_i']
            uncommon_atom_idx_j = mmp_dict['uncommon_atom_idx_j']
            
            graph_j = featurizer.tensorize(mmp_smi)
            #masked_graphs_j = featurizer.gen_masked_atom_feats(mmp_smi)
            if model.conv_name == 'gat':
                attention_j = get_attention_score(model, graph_j)            
                if torch.sum(get_uncommon_att(attention_i, uncommon_atom_idx_i)) - \
                    torch.sum(get_uncommon_att(attention_j, uncommon_atom_idx_j)) > 0:
                    gnn_direction_score['attention'].append(1)
                else:
                    gnn_direction_score['attention'].append(0)
            smoothgrad_att_j = get_smoothgrad_att(model, graph_j)
            gradcam_att_j = get_gradcam_att(model, graph_j)
            inputxgrad_att_j = get_inputxgrad_att(model, graph_j)
            #diff_mask_j = get_graph_mask_att(model, graph_j, masked_graphs_j)
            ig_att_j = get_ig_att(model, graph_j)

            if diff > 0:
                ground_truths.append(1)
            else:
                ground_truths.append(0)
            
            if torch.sum(get_uncommon_att(smoothgrad_att_i, uncommon_atom_idx_i)) - \
                    torch.sum(get_uncommon_att(smoothgrad_att_j, uncommon_atom_idx_j)) > 0:
                gnn_direction_score['smoothgrad'].append(1)
            else:
                gnn_direction_score['smoothgrad'].append(0)

            if torch.sum(get_uncommon_att(gradcam_att_i, uncommon_atom_idx_i)) - \
                    torch.sum(get_uncommon_att(gradcam_att_j, uncommon_atom_idx_j)) > 0:
                gnn_direction_score['gradcam'].append(1)
            else:
                gnn_direction_score['gradcam'].append(0)
            
            if torch.sum(get_uncommon_att(inputxgrad_att_i, uncommon_atom_idx_i)) - \
                    torch.sum(get_uncommon_att(inputxgrad_att_j, uncommon_atom_idx_j)) > 0:
                gnn_direction_score['inputxgrad'].append(1)
            else:
                gnn_direction_score['inputxgrad'].append(0)
            masked_graph_i = graph_i.clone()
            masked_graph_j = graph_j.clone()
            masked_graph_i.x[uncommon_atom_idx_i, :] = 0
            masked_graph_j.x[uncommon_atom_idx_j, :] = 0
            if torch.sum(get_graph_mask_att(model, graph_i, masked_graph_i)) - \
                    torch.sum(get_graph_mask_att(model, graph_j, masked_graph_j)) > 0:
                gnn_direction_score['mask'].append(1)
            else:
                gnn_direction_score['mask'].append(0)

            if torch.sum(get_uncommon_att(ig_att_i, uncommon_atom_idx_i)) - \
                    torch.sum(get_uncommon_att(ig_att_j, uncommon_atom_idx_j)) > 0:
                gnn_direction_score['ig'].append(1)
            else:
                gnn_direction_score['ig'].append(0)
            
            
    print("Total number of ground_truth explanations", num_pairs)
    # accuracy
    if model.conv_name == 'gat':
        attention_score = accuracy_score(ground_truths, gnn_direction_score['attention'])
        print("gnn attention direction accuracy score: {:.3f}".format(attention_score))
        attention_f1 = f1_score(ground_truths, gnn_direction_score['attention'])
        print("gnn attention direction F1 score: {:.3f}".format(attention_f1))
    
    smoothgrad_score = accuracy_score(ground_truths, gnn_direction_score['smoothgrad'])
    gradcam_score = accuracy_score(ground_truths, gnn_direction_score['gradcam'])
    inputxgrad_score = accuracy_score(ground_truths, gnn_direction_score['inputxgrad'])
    mask_score = accuracy_score(ground_truths, gnn_direction_score['mask'])
    ig_score = accuracy_score(ground_truths, gnn_direction_score['ig'])

    print("gnn SmoothGrad direction accuracy score: {:.3f}".format(smoothgrad_score))
    print("gnn GradCAM direction accuracy score: {:.3f}".format(gradcam_score))
    print("gnn InputXGrad direction accuracy score: {:.3f}".format(inputxgrad_score))
    print("gnn mask direction accuracy score: {:.3f}".format(mask_score))
    print("gnn ig direction accuracy score: {:.3f}".format(ig_score))

    # F1
    smoothgrad_f1 = f1_score(ground_truths, gnn_direction_score['smoothgrad'])
    gradcam_f1 = f1_score(ground_truths, gnn_direction_score['gradcam'])
    inputxgrad_f1 = f1_score(ground_truths, gnn_direction_score['inputxgrad'])
    mask_f1 = f1_score(ground_truths, gnn_direction_score['mask'])
    ig_f1 = f1_score(ground_truths, gnn_direction_score['ig'])
    print("gnn SmoothGrad direction F1 score: {:.3f}".format(smoothgrad_f1))
    print("gnn GradCAM direction F1 score: {:.3f}".format(gradcam_f1))
    print("gnn InputXGrad direction F1 score: {:.3f}".format(inputxgrad_f1))
    print("gnn mask direction F1 score: {:.3f}".format(mask_f1))
    print("gnn ig direction F1 score: {:.3f}".format(ig_f1))

    if model.conv_name == 'gat':
        return {
            'attention_acc': attention_score, 'attention_f1': attention_f1,
            'smoothgrad_acc': smoothgrad_score, 'smoothgrad_f1': smoothgrad_f1,
            'gradcam_acc': gradcam_score, 'gradcam_f1': gradcam_f1,
            'inputxgrad_acc': inputxgrad_score, 'inputxgrad_f1': inputxgrad_f1,
            'ig_acc': ig_score, 'ig_f1': ig_f1,
            'mask_acc': mask_score, 'mask_f1': mask_f1}, gnn_direction_score
    else:
        return {
            'gradcam_acc': gradcam_score, 'gradcam_f1': gradcam_f1,
            'smoothgrad_acc': smoothgrad_score, 'smoothgrad_f1': smoothgrad_f1,
            'inputxgrad_acc': inputxgrad_score, 'inputxgrad_f1': inputxgrad_f1,
            'ig_acc': ig_score, 'ig_f1': ig_f1,
            'mask_acc': mask_score, 'mask_f1': mask_f1}, gnn_direction_score

def evaluate_rf_explain_direction(cliff_dict, data_test, model_rf):
    smiles_test = [data_test[i].smiles for i in range(len(data_test))]
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