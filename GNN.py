from argparse import Namespace
from typing import List, Optional
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Linear as Lin
from torch.nn import ModuleList, ReLU
from torch.nn import Sequential as Seq
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
    Sequential
)

class GNN(torch.nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int = 32,
        num_layers: int = 3,
        num_classes: int = 1,
        conv_name: str = "nn",
        pool: str = "mean",
        dropout_rate: float = 0.5,
    ):
        super(GNN, self).__init__()
        (
            self.num_node_features,
            self.num_edge_features,
            self.num_classes,
            self.num_layers,
            self.hidden_dim,
            self.conv_name,
        ) = (
            num_node_features,
            num_edge_features,
            num_classes,
            num_layers,
            hidden_dim,
            conv_name,
        )
        
        self.node_emb = Lin(self.num_node_features, self.hidden_dim)
        self.edge_emb = Lin(self.num_edge_features, 2*self.num_edge_features)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.convs = ModuleList()

        for i in range(self.num_layers):
            if self.conv_name == "nn":
                conv = NNConv(
                    self.hidden_dim,
                    self.hidden_dim,
                    nn=Sequential('edge_attr', [
                        (Lin(2*self.num_edge_features, self.hidden_dim * self.hidden_dim), 'edge_attr -> edge_attr'),
                        ]),
                )
            elif self.conv_name == "gine":
                conv = GINEConv(
                    nn=Sequential(
                        Lin(self.hidden_dim, 2 * self.hidden_dim),
                        ReLU(),
                        Lin(2 * self.hidden_dim, self.hidden_dim),
                    ),
                    

                )
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
                self.edge_emb = Seq(Lin(self.num_edge_features, self.num_edge_features*2),
                                    ReLU(),
                                    Lin(self.num_edge_features*2, 1),
                                    ReLU())
                conv = GCNConv(self.hidden_dim, self.hidden_dim)
            else:
                raise ValueError(f"Unknown convolutional layer {self.conv_name}")
            
            self.convs.append(Sequential('x, edge_index, edge_attr', [
                (conv, 'x, edge_index, edge_attr -> x'), 
                BatchNorm(self.hidden_dim), 
                ReLU(),
                #self.dropout
                ]))
        
        #self.lin1 = Lin(self.hidden_dim*(1+self.num_layers), self.hidden_dim*(1+self.num_layers)//2)
        #self.lin2 = Lin(self.hidden_dim*(1+self.num_layers)//2, self.num_classes)
        self.lin1 = Lin(self.hidden_dim, self.hidden_dim)
        self.lin2 = Lin(self.hidden_dim, self.hidden_dim//2)
        self.lin3 = Lin(self.hidden_dim//2, self.num_classes)


        self.pool = pool
        if self.pool == "att":
            self.pool_fn = SAGPooling(self.hidden_dim)
            #self.pool_fn = AttentionalAggregation(
                #gate_nn=Seq(Lin(self.hidden_dim, 1)),
                #nn=Seq(Lin(self.hidden_dim, self.hidden_dim)),
            #)
        elif self.pool == "mean":
            self.pool_fn = global_mean_pool
        elif self.pool == "max":
            self.pool_fn = global_max_pool
        elif self.pool == "add":
            self.pool_fn = global_add_pool
        #elif self.pool == "mean+att":
        #    self.pool_fn = global_mean_pool
        #    self.pool_fn_ucn = AttentionalAggregation(
        #        gate_nn=Seq(Lin(self.hidden_dim, 1)),
        #        nn=Seq(Lin(self.hidden_dim, self.hidden_dim)),
        #    )
        else:
            raise ValueError(f"Unknown pool {self.pool}")
       

    def forward(self, x, edge_attr, edge_index, batch: Optional[torch.Tensor] = None, return_features = False, pred_mask: Optional[torch.Tensor] = None):
        node_x = self.get_node_reps(x, edge_attr, edge_index, batch)
        if batch is None:
            fc = getattr(torch, 'sum')
            graph_x = fc(node_x, dim=0) 
        else: 
            if self.pool == "att":
                x_downsampled, _, _, batch, perm, _ =  self.pool_fn(x = node_x, edge_index=edge_index, batch=batch)
                #graph_x = torch.zeros(node_x.size(0), x_downsampled.size(-1))
                #graph_x[perm] = x_downsampled
                graph_x = global_mean_pool(x_downsampled, batch)
            else:
                graph_x = self.pool_fn(node_x, batch)
                

        if return_features:
            return graph_x
        return self.get_pred(graph_x)

    def get_node_reps(self, x, edge_attr, edge_index, batch):
        """Returns the node embeddings just before the pooling layer."""
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        node_x = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            node_x.append(x)
        #return torch.cat(node_x, dim=-1)
        #alpha = 0.2
        #return alpha*node_x[-1] + (1.0 - alpha)*node_x[0]
        return node_x[-1]
    
    def get_pred(self, x):
        """Returns the prediction of the model on a graph embedding after the graph convolutional layers."""
        #x = self.lin1(x)
        #x = x.relu()
        #x = self.dropout(x)
        x = self.lin2(x)
        x = x.relu()
        x = self.dropout(x)
        pred = self.lin3(x)
        return pred