###################################################
######## modified from camptum implementation ########
###################################################from copy import deepcopy
from typing import Any, Callable, List, Tuple, Union
import torch
from torch import Tensor
from torch_geometric.data import Data
from captum.attr._utils.attribution import GradientAttribution
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.common import (
    _format_baseline,
    #_format_output,
    _validate_input,
    _expand_target,
)
from captum.attr._utils.approximation_methods import approximation_parameters
from captum._utils.typing import (
    BaselineType,
    TargetType,
)

def batch_edge_indices(edge_indices, num_nodes):
    """
    Batch edge indices for multiple graphs to form a single large disconnected graph.
    """
    batched_edge_index, batch = [], []
    cumulative_num_nodes = 0
    for i, edge_index in enumerate(edge_indices):
        offset_edge_index = edge_index + cumulative_num_nodes
        batched_edge_index.append(offset_edge_index)
        cumulative_num_nodes += num_nodes
        batch.append([i]*num_nodes)
    batched_edge_index = torch.cat(batched_edge_index, dim=1)
    batch = torch.tensor(batch).reshape(-1)
    return (batched_edge_index, batch)

def _reshape_and_sum(
    tensor_input: Tensor, num_steps: int, num_examples: int, layer_size: Tuple[int, ...]
) -> Tensor:
    # Used for attribution methods which perform integration
    # Sums across integration steps by reshaping tensor to
    # (num_steps, num_examples, (layer_size)) and summing over
    # dimension 0. Returns a tensor of size (num_examples, (layer_size))
    return torch.sum(
        tensor_input.reshape((num_steps, num_examples) + layer_size), dim=0
    )

class IntegratedGradient(GradientAttribution):
    def __init__(
        self, 
        forward_func: Callable,
        multiply_by_inputs: bool = True):
        super(IntegratedGradient, self).__init__(forward_func)
        self.multiply_by_inputs = multiply_by_inputs

    def attribute(
        self,
        x: Tensor,
        edge_attr: Tensor,
        baselines: Tuple[Union[Tensor, int, float], ...],
        target: TargetType = None,
        edge_index: Tensor = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        step_sizes_and_alphas: Union[None, Tuple[List[float], List[float]]] = None,
    ) -> Tuple[Tensor, ...]:
        inputs = (x, edge_attr)
        baselines = _format_baseline(baselines, inputs)
        #_validate_input(inputs, baselines, n_steps, method)
        if step_sizes_and_alphas is None:
            # retrieve step size and scaling factor for specified
            # approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas
        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        input_additional_args = batch_edge_indices([edge_index] * n_steps, x.shape[0])
        expanded_target = _expand_target(target, n_steps)
        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = self.gradient_func(
            forward_fn=self.forward_func,
            inputs=scaled_features_tpl,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
        )
        # flattening grads so that we can multilpy it with step-size
        # calling contiguous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).float().view(n_steps, 1).to(grad.device)
            for grad in grads
        ]

        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimensionality as inputs
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        )
        # computes attribution for each tensor in input tuple
        # attributions has the same dimensionality as inputs
        if not self.multiply_by_inputs:
            attributions = total_grads
        else:
            attributions = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(total_grads, inputs, baselines)
            )
        return attributions
