from argparse import Namespace
from typing import List, Optional
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import ModuleList, ReLU, Linear as Lin, Sequential as Seq, Sigmoid
from torch_geometric.nn import (
    BatchNorm,
    GATConv,
    GENConv,
    GINEConv,
    NNConv,
    GCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    SAGPooling,
    Sequential,
    MLP
)
from torch_scatter import scatter
from captum._utils.gradient import _forward_layer_distributed_eval
from XACs.utils.explain_utils import process_layer_gradients_and_eval

class GNN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int = 42,
        num_edge_features: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 1,
        conv_name: str = "nn",
        pool: str = "mean",
        dropout_rate: float = 0.5,
        attribute_to_last_layer: bool = True,
        normalize_att: bool = False,
    ):
        super(GNN, self).__init__()
        (
            self.num_node_features,
            self.num_edge_features,
            self.num_classes,
            self.num_layers,
            self.hidden_dim,
            self.conv_name,
            self.dropout,
            self.attribute_to_last_layer,
            self.normalize_att,
        ) = (
            num_node_features,
            num_edge_features,
            num_classes,
            num_layers,
            hidden_dim,
            conv_name,
            dropout_rate,
            attribute_to_last_layer,
            normalize_att,
        )
        self.node_emb = Lin(self.num_node_features, self.hidden_dim)
        self.edge_emb = Lin(self.num_edge_features, self.hidden_dim)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.convs = ModuleList()

        for i in range(self.num_layers):
            if self.conv_name == "nn":
                conv = NNConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    nn=Sequential('edge_attr', [
                        (Lin(self.hidden_dim, self.hidden_dim * self.hidden_dim), 'edge_attr -> edge_attr'),
                        ]),
                )
            elif self.conv_name == "gine":
                conv = GINEConv(Seq(Lin(self.hidden_dim, 2*self.hidden_dim),
                                    ReLU(), 
                                    Lin(2*self.hidden_dim, self.hidden_dim)),
                            train_eps=True)
            elif self.conv_name == "gat":
                conv = GATConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    edge_dim=self.hidden_dim,
                    concat=False,
                )
            elif self.conv_name == "gen":
                conv = GENConv(self.hidden_dim, self.hidden_dim)
            elif self.conv_name == "gcn":
                # transform edge features to [num_edge]
                self.edge_emb = Seq(Lin(self.num_edge_features, self.hidden_dim),
                                    ReLU(),
                                    Lin(self.hidden_dim2, 1),
                                    ReLU())
                conv = GCNConv(self.hidden_dim, self.hidden_dim)
            else:
                raise ValueError(f"Unknown convolutional layer {self.conv_name}")
            
            self.convs.append(Sequential('x, edge_index, edge_attr', [
                (conv, 'x, edge_index, edge_attr -> x'), 
                #BatchNorm(self.hidden_dim), 
                ReLU(),
                #self.dropout
                ]))

        self.pool = pool
        if self.pool == "mean":
            self.pool_fn = global_mean_pool
        elif self.pool == "max":
            self.pool_fn = global_max_pool
        elif self.pool == "add":
            self.pool_fn = global_add_pool
        else:
            raise ValueError(f"Unknown pool {self.pool}")
       
        self.node_ws = ModuleList()
        if attribute_to_last_layer:
            self.node_ws.append(Seq(Lin(self.hidden_dim, 1), Sigmoid()))
        else:
            for i in range(self.num_layers):
                self.node_ws.append(Seq(Lin(self.hidden_dim, 1), Sigmoid()))

        self.mlp = MLP([self.hidden_dim, self.hidden_dim//2, num_classes], dropout=dropout_rate, norm=None)

    def forward(self, x, edge_attr, edge_index, batch: Optional[torch.Tensor] = None):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
        x = self.pool_fn(x, batch)
        x = self.mlp(x)
        return x

    def explanation_forward(self, batched_data):
        (batched_x, 
         batched_edge_index,
         batched_edge_attr,
         graph_mask,
         mini_batch,
         uncom_mask,
         common_mask,
         batched_batch) = (batched_data.x,
            batched_data.edge_index,
            batched_data.edge_attr,
            batched_data.graph_mask,
            batched_data.mini_batch,
            batched_data.uncom_atom_mask,
            batched_data.common_atom_mask,
            batched_data.batch)

        saved_layer, output = _forward_layer_distributed_eval(
            self.forward,
            (batched_x, batched_edge_attr),
            self.convs[-1] if self.attribute_to_last_layer else [self.convs[i] for i in range(self.num_layers)],
            target_ind=None,
            additional_forward_args=(batched_edge_index, mini_batch),
            attribute_to_layer_input=False,
            forward_hook_with_return=True,
            require_layer_grads=True,
        )

        layer_gradients, layer_evals = process_layer_gradients_and_eval(
            saved_layer, 
            output,
            self.forward, self.convs[-1] if self.attribute_to_last_layer 
            else [self.convs[i] for i in range(self.num_layers)]
        )
        #for i, (layer_evals, layer_gradients) in enumerate(zip(layer_evals, layer_gradients)):
            #att = torch.mul(layer_evals, layer_gradients)
            #W = self.node_ws[i](att)
            #att = torch.sum(att, dim=tuple(d for d in range(1, len(att.shape))), keepdim=True)

        # this implementation does not do pooling for gradient 
        att = tuple(torch.einsum("ij, ij -> i", layer_evals, layer_gradients) 
                    for layer_evals, layer_gradients in zip(layer_evals, layer_gradients))[0].reshape(-1, 1)

        # this implementation does average pooling for gradient; the average is done feature wise making every node a weight
        #pool_grad = tuple(torch.mean(layer_gradients, dim=1) for layer_gradients in layer_gradients)[0].reshape(-1, 1)
        #att = tuple(torch.einsum("ij, i -> i", layer_evals, pool_grad) 
                    #for layer_evals, pool_grad in zip(layer_evals, pool_grad))[0].reshape(-1, 1)

        # this implementation does average pooling for gradient; the average is done batch wise making every feature a weight
        #pool_grad = tuple(torch.mean(layer_gradients, dim=0) for layer_gradients in layer_gradients)[0].reshape(-1, 1)
        #att = tuple(torch.einsum("ij, j -> i", layer_evals, pool_grad) 
                    #for layer_evals, pool_grad in zip(layer_evals, pool_grad))[0].reshape(-1, 1)

        # this implementation does average pooling for gradient; the average is done molecule wise making every feature a weight
        #pool_grad = tuple(scatter(layer_gradients, batched_batch, dim=0, reduce='mean') for layer_gradients in layer_gradients)[0].reshape(-1, 1)
        #att = tuple(torch.einsum("ij, j -> i", layer_evals, pool_grad) 
                    #for layer_evals, pool_grad in zip(layer_evals, pool_grad))[0].reshape(-1, 1)
        if self.normalize_att:
            att = (att - att.min()) / (att.max() - att.min())
        
        uncom_att, common_att = att*uncom_mask, att*common_mask
        sum_uncom_att = scatter(uncom_att, batched_batch, dim=0, reduce='add')
        #sum_common_att = scatter(common_att, batched_batch, dim=0, reduce='add')

        assert graph_mask.shape[0] == output.shape[0], "graph mask and output shape mismatch"
        output = output.reshape(-1, 1)[graph_mask==1.]
        return output, sum_uncom_att, common_att