###################################################
######## modified from DIGS implementation ########
###################################################
from typing import Any, Callable, List, Tuple, Union
from captum._utils.typing import TargetType
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import captum.attr as ca
from captum._utils.common import (
    _format_additional_forward_args,
    #_format_output,
    _format_inputs,
    _is_tuple,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from explain.explain_utils import compute_layer_gradients_and_eval

class GraphLayerGradCam(ca.LayerGradCam):

    def __init__(
            self,
            forward_func: Callable,
            layer: Module,
            device_ids: Union[None, List[int]] = None,
            training: bool = False,
    ) -> None:
        super().__init__(forward_func, layer, device_ids)
        self.training = training

    def attribute(
            self,
            inputs: Union[Tensor, Tuple[Tensor, ...]],
            target: TargetType = None,
            additional_forward_args: Any = None,
            attribute_to_layer_input: bool = False,
            return_gradients: bool = False,
            relu_attributions: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        inputs = _format_inputs(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs, warn=False)
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        #is_layer_tuple = isinstance(layer_gradients, tuple)
        undo_gradient_requirements(inputs, gradient_mask)
        #print('layer_gradients', layer_gradients)
        #print('layer_evals', layer_evals)
        # Gradient Calculation end

        ## Addition: shape from PyG to General PyTorch
        # The default implementation from DIGS shapes the layer_grad/layer_evels
        # from [num_nodes, feature_size] to [1, feature_size, num_nodes].
        # The gradient will then be averaged over the num_nodes dimension,
        layer_gradients = tuple(layer_grad.transpose(0, 1).unsqueeze(0)
                                for layer_grad in layer_gradients)

        layer_evals = tuple(layer_eval.transpose(0, 1).unsqueeze(0)
                            for layer_eval in layer_evals)
        
        # end
        summed_grads = tuple(
            torch.mean(
                layer_grad,
                dim=tuple(x for x in range(2, len(layer_grad.shape))),
                keepdim=True,
            )
            for layer_grad in layer_gradients
        )

        if return_gradients:
            scaled_grads = tuple(
                torch.sum(summed_grad, dim=1, keepdim=True)
                for summed_grad in summed_grads
            )
            scaled_grads = tuple(scaled_grad.squeeze(0).transpose(0, 1)
                                 for scaled_grad in scaled_grads)
            return scaled_grads[0] if _is_tuple(scaled_grads) else scaled_grads
        scaled_acts = tuple(
            torch.sum(summed_grad * layer_eval, dim=1, keepdim=True)
            for summed_grad, layer_eval in zip(summed_grads, layer_evals)
        )
        if relu_attributions:
            scaled_acts = tuple(F.relu(scaled_act) for scaled_act in scaled_acts)

        # what I add: shape from General PyTorch to PyG

        scaled_acts = tuple(scaled_act.squeeze(0).transpose(0, 1)
                            for scaled_act in scaled_acts)

        # end

        # only returns the first element of the tuple, which is the node attribution
        return scaled_acts[0] if _is_tuple(scaled_acts) else scaled_acts