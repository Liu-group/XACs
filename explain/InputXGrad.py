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
import torch
from torch import Tensor




class InputXGradient(GradientAttribution):
    def __init__(self, forward_func: Callable):
        GradientAttribution.__init__(self, forward_func)

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
        gradients = self.gradient_func(
            self.forward_func,
            inputs,
            target,
            additional_forward_args,
        )
        attributions = tuple(
            input * gradient for input, gradient in zip(inputs, gradients)
        )
        undo_gradient_requirements(inputs, gradient_mask)
        return attributions if _is_tuple(inputs) else attributions[0]
        