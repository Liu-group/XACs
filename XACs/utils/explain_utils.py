######################################
######## modified from caputm ########  
######################################
from typing import Any, Dict, Callable, cast, List, Tuple, Union
from captum._utils.common import (
    _reduce_list,
    _run_forward,
    _sort_key_list,
)
from captum._utils.typing import (
    ModuleOrModuleList,
    TargetType,
)
from torch import Tensor, device
from torch.nn import Module
from captum._utils.gradient import (
    _forward_layer_distributed_eval,
    _neuron_gradients,
    _extract_device_ids,
)
import torch   

def compute_gradients(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
) -> Tuple[Tensor, ...]:
    r"""
    Computes gradients of the output with respect to inputs for an
    arbitrary forward function.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        input:      Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        additional_forward_args: Additional input arguments that forward
                    function requires. It takes an empty tuple (no additional
                    arguments) if no additional arguments are required
    """
    with torch.autograd.set_grad_enabled(True):
        # runs forward pass
        outputs = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
        assert outputs[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )
        # torch.unbind(forward_out) is a list of scalar tensor tuples and
        # contains batch_size * #steps elements
        grads = torch.autograd.grad(torch.unbind(outputs), inputs, create_graph=True)
    return grads

def compute_layer_gradients_and_eval(
    forward_fn: Callable,
    layer: ModuleOrModuleList,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
    gradient_neuron_selector: Union[
        None, int, Tuple[Union[int, slice], ...], Callable
    ] = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    output_fn: Union[None, Callable] = None,
) -> Union[
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[List[Tuple[Tensor, ...]], List[Tuple[Tensor, ...]]],
]:
    r"""
    Computes gradients of the output with respect to a given layer as well
    as the output evaluation of the layer for an arbitrary forward function
    and given input.

    For data parallel models, hooks are executed once per device ,so we
    need to internally combine the separated tensors from devices by
    concatenating based on device_ids. Any necessary gradients must be taken
    with respect to each independent batched tensor, so the gradients are
    computed and combined appropriately.

    More information regarding the behavior of forward hooks with DataParallel
    models can be found in the PyTorch data parallel documentation. We maintain
    the separate inputs in a dictionary protected by a lock, analogous to the
    gather implementation for the core PyTorch DataParallel implementation.

    NOTE: To properly handle inplace operations, a clone of the layer output
    is stored. This structure inhibits execution of a backward hook on the last
    module for the layer output when computing the gradient with respect to
    the input, since we store an intermediate clone, as
    opposed to the true module output. If backward module hooks are necessary
    for the final module when computing input gradients, utilize
    _forward_layer_eval_with_neuron_grads instead.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        layer:      Layer for which gradients / output will be evaluated.
        inputs:     Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        output_fn:  An optional function that is applied to the layer inputs or
                    outputs depending whether the `attribute_to_layer_input` is
                    set to `True` or `False`
        args:       Additional input arguments that forward function requires.
                    It takes an empty tuple (no additional arguments) if no
                    additional arguments are required


    Returns:
        tuple[**gradients**, **evals**]:
        - **gradients**:
            Gradients of output with respect to target layer output.
        - **evals**:
            Target layer output for given input.
    """
    with torch.autograd.set_grad_enabled(True):
        # saved_layer is a dictionary mapping device to a tuple of
        # layer evaluations on that device.
        saved_layer, output = _forward_layer_distributed_eval(
            forward_fn,
            inputs,
            layer,
            target_ind=target_ind,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
            forward_hook_with_return=True,
            require_layer_grads=True,
        )
        assert output[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )

        device_ids = _extract_device_ids(forward_fn, saved_layer, device_ids)

        # Identifies correct device ordering based on device ids.
        # key_list is a list of devices in appropriate ordering for concatenation.
        # If only one key exists (standard model), key list simply has one element.
        key_list = _sort_key_list(
            list(next(iter(saved_layer.values())).keys()), device_ids
        )
        all_outputs: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]
        if isinstance(layer, Module):
            all_outputs = _reduce_list(
                [
                    saved_layer[layer][device_id]
                    if output_fn is None
                    else output_fn(saved_layer[layer][device_id])
                    for device_id in key_list
                ]
            )
        else:
            all_outputs = [
                _reduce_list(
                    [
                        saved_layer[single_layer][device_id]
                        if output_fn is None
                        else output_fn(saved_layer[single_layer][device_id])
                        for device_id in key_list
                    ]
                )
                for single_layer in layer
            ]
        all_layers: List[Module] = [layer] if isinstance(layer, Module) else layer
        grad_inputs = tuple(
            layer_tensor
            for single_layer in all_layers
            for device_id in key_list
            for layer_tensor in saved_layer[single_layer][device_id]
        )
        ## Modify the following line to use the let graph be created
        saved_grads = torch.autograd.grad(torch.unbind(output), grad_inputs, create_graph=True)
        offset = 0
        all_grads: List[Tuple[Tensor, ...]] = []
        for single_layer in all_layers:
            num_tensors = len(next(iter(saved_layer[single_layer].values())))
            curr_saved_grads = [
                saved_grads[i : i + num_tensors]
                for i in range(
                    offset, offset + len(key_list) * num_tensors, num_tensors
                )
            ]
            offset += len(key_list) * num_tensors
            if output_fn is not None:
                curr_saved_grads = [
                    output_fn(curr_saved_grad) for curr_saved_grad in curr_saved_grads
                ]

            all_grads.append(_reduce_list(curr_saved_grads))

        layer_grads: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]
        layer_grads = all_grads
        if isinstance(layer, Module):
            layer_grads = all_grads[0]

        if gradient_neuron_selector is not None:
            assert isinstance(
                layer, Module
            ), "Cannot compute neuron gradients for multiple layers simultaneously!"
            inp_grads = _neuron_gradients(
                inputs, saved_layer[layer], key_list, gradient_neuron_selector
            )
            return (
                cast(Tuple[Tensor, ...], layer_grads),
                cast(Tuple[Tensor, ...], all_outputs),
                inp_grads,
            )
    return layer_grads, all_outputs  # type: ignore

def process_layer_gradients_and_eval(
    saved_layer: Dict[Module, Dict[int, Tuple[Tensor, ...]]],
    output: Dict[Module, Dict[device, Tuple[Tensor, ...]]],
    forward_fn: Callable,
    layer: ModuleOrModuleList,
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    output_fn: Union[None, Callable] = None,
) -> Union[
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[List[Tuple[Tensor, ...]], List[Tuple[Tensor, ...]]],
]:
    r"""
    Modified version of compute_layer_gradients_and_eval from captum.
    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        layer:      Layer for which gradients / output will be evaluated.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        output_fn:  An optional function that is applied to the layer inputs or
                    outputs depending whether the `attribute_to_layer_input` is
                    set to `True` or `False`
        args:       Additional input arguments that forward function requires.
                    It takes an empty tuple (no additional arguments) if no
                    additional arguments are required


    Returns:
        tuple[**gradients**, **evals**]:
        - **gradients**:
            Gradients of output with respect to target layer output.
        - **evals**:
            Target layer output for given input.
    """
    with torch.autograd.set_grad_enabled(True):
        device_ids = _extract_device_ids(forward_fn, saved_layer, device_ids)

        # Identifies correct device ordering based on device ids.
        # key_list is a list of devices in appropriate ordering for concatenation.
        # If only one key exists (standard model), key list simply has one element.
        key_list = _sort_key_list(
            list(next(iter(saved_layer.values())).keys()), device_ids
        )
        all_outputs: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]
        if isinstance(layer, Module):
            all_outputs = _reduce_list(
                [
                    saved_layer[layer][device_id]
                    if output_fn is None
                    else output_fn(saved_layer[layer][device_id])
                    for device_id in key_list
                ]
            )
        else:
            all_outputs = [
                _reduce_list(
                    [
                        saved_layer[single_layer][device_id]
                        if output_fn is None
                        else output_fn(saved_layer[single_layer][device_id])
                        for device_id in key_list
                    ]
                )
                for single_layer in layer
            ]
        all_layers: List[Module] = [layer] if isinstance(layer, Module) else layer
        grad_inputs = tuple(
            layer_tensor
            for single_layer in all_layers
            for device_id in key_list
            for layer_tensor in saved_layer[single_layer][device_id]
        )
        ## Modify the following line to let graph be created
        saved_grads = torch.autograd.grad(torch.unbind(output), grad_inputs, create_graph=True)
        offset = 0
        all_grads: List[Tuple[Tensor, ...]] = []
        for single_layer in all_layers:
            num_tensors = len(next(iter(saved_layer[single_layer].values())))
            curr_saved_grads = [
                saved_grads[i : i + num_tensors]
                for i in range(
                    offset, offset + len(key_list) * num_tensors, num_tensors
                )
            ]
            offset += len(key_list) * num_tensors
            if output_fn is not None:
                curr_saved_grads = [
                    output_fn(curr_saved_grad) for curr_saved_grad in curr_saved_grads
                ]

            all_grads.append(_reduce_list(curr_saved_grads))

        layer_grads: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]
        layer_grads = all_grads
        if isinstance(layer, Module):
            layer_grads = all_grads[0]

    return layer_grads, all_outputs  # type: ignore