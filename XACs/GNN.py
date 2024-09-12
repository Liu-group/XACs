from argparse import Namespace
from typing import List, Optional
import torch
from torch.nn import ModuleList, ReLU, Linear as Lin, Sequential as Seq, Sigmoid
from torch_geometric.nn import (
    BatchNorm,
    GATv2Conv,
    GENConv,
    GINEConv,
    NNConv,
    GCNConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    Set2Set,
    SAGPooling,
    Sequential,
    MLP
)
from torch_scatter import scatter
from captum._utils.gradient import _forward_layer_distributed_eval
from XACs.utils.explain_utils import process_layer_gradients_and_eval
from XACs.utils.ifm import PLE

class GNN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int = 42,
        num_edge_features: int = 6,
        node_hidden_dim: int = 128,
        edge_hidden_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 1,
        conv_name: str = "nn",
        pool: str = "mean",
        num_layer_set2set: int = 3,
        heads: int = 8,
        dropout_rate: float = 0.5,
        attribute_to_last_layer: bool = True,
        normalize_att: bool = False,
        uncom_pool: str = 'add',
        ifp: bool = False,
    ):
        super(GNN, self).__init__()
        self.conv_name = conv_name
        self.num_layers = num_layers
        self.attribute_to_last_layer = attribute_to_last_layer
        self.normalize_att = normalize_att
        self.uncom_pool = uncom_pool
        if ifp:
            node_ifp = PLE(num_node_features, node_hidden_dim)
            edge_ifp = PLE(num_edge_features, node_hidden_dim)
            node_lin = Lin(node_hidden_dim*2, node_hidden_dim)
            edge_lin = Lin(edge_hidden_dim*2, edge_hidden_dim)
            self.node_emb = Seq(node_ifp, node_lin, ReLU())
            self.edge_emb = Seq(edge_ifp, edge_lin, ReLU())
            print("Using IFP")
        else:
            self.node_emb = Lin(num_node_features, node_hidden_dim)
            self.edge_emb = Lin(num_edge_features, edge_hidden_dim)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.convs = ModuleList()

        for i in range(self.num_layers):
            if conv_name == "nn":
                conv = NNConv(
                    in_channels=node_hidden_dim if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    nn=Sequential('edge_attr', [
                        (Lin(edge_hidden_dim, node_hidden_dim*edge_hidden_dim), 'edge_attr -> edge_attr'),
                        ]),
                )
            elif conv_name == "gine":
                conv = GINEConv(Seq(Lin(node_hidden_dim if i == 0 else hidden_dim, 2*hidden_dim),
                                    ReLU(), 
                                    Lin(2*hidden_dim, hidden_dim)),
                                edge_dim=edge_hidden_dim,
                                train_eps=True)
            elif conv_name == "gat":
                conv = GATv2Conv(
                    in_channels=node_hidden_dim if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    edge_dim=edge_hidden_dim,
                    concat=False,
                )
            elif conv_name == "gcn":
                self.edge_emb = Seq(Lin(num_edge_features, hidden_dim),
                                    ReLU(),
                                    Lin(hidden_dim2, 1),
                                    ReLU())
                conv = GCNConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown convolutional layer {conv_name}")
            
            if conv_name == "gat":
                self.convs.append(conv)
            else:
                self.convs.append(Sequential('x, edge_index, edge_attr', [
                    (conv, 'x, edge_index, edge_attr -> x'), 
                    #BatchNorm(hidden_dim), 
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
        elif self.pool == 'set2set':
            self.pool_fn = Set2Set(hidden_dim, processing_steps=num_layer_set2set)
        else:
            raise ValueError(f"Unknown pool {self.pool}")
       
        self.node_ws = ModuleList()
        if attribute_to_last_layer:
            self.node_ws.append(Seq(Lin(hidden_dim, 1), Sigmoid()))
        else:
            for i in range(self.num_layers):
                self.node_ws.append(Seq(Lin(hidden_dim, 1), Sigmoid()))

        self.mlp = MLP([hidden_dim, hidden_dim//2, num_classes], dropout=dropout_rate, norm=None)

    def forward(self, x, edge_attr, edge_index, batch: Optional[torch.Tensor] = None, **kwargs):
        return_attention_weights = kwargs.get('return_attention_weights', False)
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        atts = []
        for i in range(self.num_layers):
            if self.conv_name == "gat":
                if return_attention_weights:
                    x, att = self.convs[i](x, edge_index, edge_attr, return_attention_weights=return_attention_weights) 
                    atts.append(att)
                else:
                    x = self.convs[i](x, edge_index, edge_attr)
                x = x.relu()
            else:        
                x = self.convs[i](x, edge_index, edge_attr)
        x = self.pool_fn(x, batch)
        x = self.mlp(x)
        return x if not return_attention_weights else (x, atts)

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
        # this implementation does not do pooling for gradient 
        att = tuple(torch.einsum("ij, ij -> i", layer_eval, layer_gradient) 
                    for layer_eval, layer_gradient in zip(layer_evals, layer_gradients))[0].reshape(-1, 1)
        if self.normalize_att:
            att = (att - att.min()) / (att.max() - att.min())
        uncom_att, common_att = att*uncom_mask, att*common_mask
        if self.uncom_pool == 'add':
            pooled_uncom_att_diff = scatter(uncom_att, batched_batch, dim=0, reduce='add')
        else: # mean
            num_uncom = scatter(uncom_mask, mini_batch, dim=0, reduce='add')
            pooled_uncom_att = scatter(uncom_att, mini_batch, dim=0, reduce='add') / (num_uncom + 1e-6)
            bridge_batch = scatter(batched_batch, mini_batch, dim=0, reduce='mean')
            pooled_uncom_att_diff = scatter(pooled_uncom_att, bridge_batch, dim=0, reduce='add')
        assert graph_mask.shape[0] == output.shape[0], "graph mask and output shape mismatch"
        output = output.reshape(-1, 1)[graph_mask==1.]
        return output, pooled_uncom_att_diff, common_att
    
    def gnes_forward(self, x, edge_attr, edge_index, batch: Optional[torch.Tensor] = None):
        saved_layer, output = _forward_layer_distributed_eval(
            self.forward,
            (x, edge_attr),
            self.convs[-1],
            target_ind=None,
            additional_forward_args=(edge_index, batch),
            attribute_to_layer_input=False,
            forward_hook_with_return=True,
            require_layer_grads=True,
        )

        layer_gradients, layer_evals = process_layer_gradients_and_eval(
            saved_layer, 
            output,
            self.forward, self.convs[-1]
        )
        layer_gradient, layer_eval = layer_gradients[0], layer_evals[0]
        att = scatter(layer_gradient, batch, dim=0, reduce='mean')[batch]
        #att = torch.einsum("ij, ij -> i", layer_eval, layer_gradient).reshape(-1, 1)
        return output, att