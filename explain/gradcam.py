from typing import Any, Callable, List, Tuple, Union
from captum._utils.typing import TargetType
import captum.attr as ca
from captum.attr._utils.attribution import GradientAttribution
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from torch.nn import Module
from captum._utils.common import (
    _format_additional_forward_args,
    #_format_output,
    _format_inputs,
    _is_tuple,
)
# add ./explain/ to sys.path
import sys
sys.path.append('./')
from explain_utils import compute_layer_gradients_and_eval

import torch
from torch import Tensor




class GradCAM(ca.LayerGradCam):
    def __init__(self, 
                 forward_func: Callable,
                 layer: Module,
                 device_ids: Union[None, List[int]] = None,
                 training: bool = False):
        super().__init__(forward_func, layer, device_ids)

    def attribute(
            self,
            inputs: Union[Tensor, Tuple[Tensor, ...]],
            target: TargetType = None,
            additional_forward_args: Any = None,
            attribute_to_layer_input: bool = False,
            return_gradients: bool = False,
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
        undo_gradient_requirements(inputs, gradient_mask)
        print("layer_evals", layer_evals)
        print("layer_evals[0]", layer_evals[0].shape)
        print("layer_evals[1]", layer_evals[1].shape)
        print("layer_gradients", layer_gradients)
        print("layer_gradients[0]", layer_gradients[0].shape)
        print("layer_gradients[1]", layer_gradients[1].shape)
        attributions = (
                torch.einsum("ij, ij -> i", layer_evals, layer_gradients)
            )

        return attributions