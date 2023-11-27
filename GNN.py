from argparse import Namespace
from typing import List, Optional
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import ModuleList, ReLU, Linear as Lin, Sequential as Seq
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
                BatchNorm(self.hidden_dim), 
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
       
        self.mlp = MLP([self.hidden_dim, self.hidden_dim//2, num_classes], dropout=dropout_rate, norm=None)

    def forward(self, x, edge_attr, edge_index, batch: Optional[torch.Tensor] = None):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
        x = self.pool_fn(x, batch)
        x = self.mlp(x)
          
        return x