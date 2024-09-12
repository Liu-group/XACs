# Code adopted from: https://github.com/junxia97/IFM/blob/18fe184eb44a761bd69d9a9b238543ca8147180b/egnn.py#L291
# "Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions"
import torch
import torch.nn as nn
import numpy as np

class PLE(nn.Module):
    def __init__(self, n_num_features: int, d_out: int, sigma: float = 1.) -> None:
        super().__init__()
        self.d_out = d_out
        self.sigma = sigma
        coefficients = torch.Tensor(n_num_features, d_out)
        self.coefficients = nn.Parameter(coefficients)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.normal_(self.coefficients, 0.0, self.sigma)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = 2*np.pi*self.coefficients[None]*x[..., None]
        x = 2*np.pi*torch.matmul(x, self.coefficients)

        return torch.cat([torch.cos(x), torch.sin(x)], -1)