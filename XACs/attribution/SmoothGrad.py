###################################################
######## modified from DIGS GradInput implementation ########
###################################################
from typing import Any, Callable, List, Tuple, Union
from captum._utils.typing import TargetType
from captum.attr._utils.attribution import GradientAttribution
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.common import (
    _format_additional_forward_args,
    #_format_output,
    _format_inputs,
    _is_tuple,
)
from XACs.utils.explain_utils import compute_gradients
import torch
from torch import Tensor

class SmoothGrad(GradientAttribution):
    def __init__(self, 
                forward_func: Callable, 
                noise_var: float = 0.15,
                n_samples: int = 50,
                training: bool = False,
                ):
        GradientAttribution.__init__(self, forward_func)
        if training:
            self.gradient_func = compute_gradients
        self.noise_var = noise_var
        self.n_samples = n_samples

    def attribute(
            self,
            inputs: Union[Tensor, Tuple[Tensor, ...]],
            target: TargetType = None,
            additional_forward_args: Any = None,
            attribute_to_layer_input: bool = False,
            return_gradients: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:        
        inputs = _format_inputs(inputs)
        additional_forward_args = _format_additional_forward_args(additional_forward_args)  
        for i in range(self.n_samples):
            noises = tuple(torch.normal(mean=0, std=self.noise_var, size=input.size()) for input in inputs)
            noisy_inputs = tuple((input + noise) for input, noise in zip(inputs, noises))
            gradient_mask = apply_gradient_requirements(noisy_inputs, warn=False)
            gradients = self.gradient_func(
                self.forward_func,
                noisy_inputs,
                target,
                additional_forward_args,
            )
            attributions = tuple(
                torch.einsum("ij, ij -> i", noisy_input, gradient) for noisy_input, gradient in zip(noisy_inputs, gradients)
            )
            if i == 0:
                total_attributions = attributions
            else:
                total_attributions = tuple(
                    total_attribution + attribution for total_attribution, attribution in zip(total_attributions, attributions)
                )
            undo_gradient_requirements(noisy_inputs, gradient_mask)
        attributions = tuple(
            total_attribution / self.n_samples for total_attribution in total_attributions
        )
        return attributions if _is_tuple(inputs) else attributions[0]
        