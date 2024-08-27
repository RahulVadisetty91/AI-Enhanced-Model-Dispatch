
import torch
from torch import nn
import torch.cuda.comm
import copy
import gc
import os
import sys
import itertools
import bisect
import random
import utils
from typing import Dict, List, Optional, Union

from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging

logger = logging.get_logger(__name__)

# AI-driven configuration and dynamic updates
def configure_dynamic_settings(device_map: Dict[str, Union[str, int, torch.device]], model: nn.Module):
    """
    Dynamically configure device settings for model deployment based on available resources and AI-driven recommendations.
    """
    # AI-driven recommendations for optimal device configuration
    recommended_devices = {name: torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") 
                           for name in device_map}
    # Update the device map with recommendations
    device_map.update(recommended_devices)
    logger.info(f"Device configuration updated: {device_map}")

    return device_map

# Enhanced error handling function
def handle_error(error_message: str):
    logger.error(f"Error encountered: {error_message}")
    # Implement AI-driven error recovery strategies if applicable
    # For now, we'll just raise the error
    raise RuntimeError(error_message)

breakmodel = True
gpu_blocks = []
disk_blocks = 0
primary_device = 0 if torch.cuda.device_count() > 0 else "cpu"

if utils.HAS_ACCELERATE:
    from accelerate.hooks import attach_align_device_hook_on_blocks
    from accelerate.utils import OffloadedWeightsLoader, check_device_map, extract_submodules_state_dict, offload_state_dict
    from accelerate import dispatch_model

def dispatch_model_ex(
    model: nn.Module,
    device_map: Dict[str, Union[str, int, torch.device]],
    main_device: Optional[torch.device] = None,
    state_dict: Optional[Dict[str, torch.Tensor]] = None,
    offload_dir: Union[str, os.PathLike] = None,
    offload_buffers: bool = False,
    **kwargs,
):
    """
    This is a modified version of
    https://github.com/huggingface/accelerate/blob/eeaba598f455fbd2c48661d7e816d3ff25ab050b/src/accelerate/big_modeling.py#L130
    that still works when the main device is the CPU.

    Dispatches a model according to a given device map. Layers of the model might be spread across GPUs, offloaded on
    the CPU or even the disk.

    Args:
        model (`torch.nn.Module`):
            The model to dispatch.
        device_map (`Dict[str, Union[str, int, torch.device]]`):
            A dictionary mapping module names in the models `state_dict` to the device they should go to. Note that
            `"disk"` is accepted even if it's not a proper value for `torch.device`.
        main_device (`str`, `int` or `torch.device`, *optional*):
            The main execution device. Will default to the first device in the `device_map` different from `"cpu"` or
            `"disk"`.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the part of the model that will be kept on CPU.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
    """
    try:
        if main_device != "cpu":
            return dispatch_model(model, device_map, main_device, state_dict, offload_dir=offload_dir, offload_buffers=offload_buffers, **kwargs)

        # Configure devices dynamically
        device_map = configure_dynamic_settings(device_map, model)

        # Error early if the device map is incomplete.
        check_device_map(model, device_map)

        offload_devices = ["cpu", "disk"] if main_device != "cpu" else ["disk"]

        if main_device is None:
            main_device = [d for d in device_map.values() if d not in offload_devices][0]

        cpu_modules = [name for name, device in device_map.items() if device == "cpu"] if main_device != "cpu" else []
        if state_dict is None and len(cpu_modules) > 0:
            state_dict = extract_submodules_state_dict(model.state_dict(), cpu_modules)

        disk_modules = [name for name, device in device_map.items() if device == "disk"]
        if offload_dir is None and len(disk_modules) > 0:
            raise ValueError(
                "We need an `offload_dir` to dispatch this model according to this `device_map`, the following submodules "
                f"need to be offloaded: {', '.join(disk_modules)}."
            )
        if len(disk_modules) > 0 and (
            not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, "index.json"))
        ):
            disk_state_dict = extract_submodules_state_dict(model.state_dict(), disk_modules)
            offload_state_dict(offload_dir, disk_state_dict)

        execution_device = {
            name: main_device if device in offload_devices else device for name, device in device_map.items()
        }
        offload = {name: device in offload_devices for name, device in device_map.items()}
        save_folder = offload_dir if len(disk_modules) > 0 else None
        if state_dict is not None or save_folder is not None:
            weights_map = OffloadedWeightsLoader(state_dict=state_dict, save_folder=save_folder)
        else:
            weights_map = None

        attach_align_device_hook_on_blocks(
            model,
            execution_device=execution_device,
            offload=offload,
            offload_buffers=offload_buffers,
            weights_map=weights_map,
            **kwargs,
        )
        model.hf_device_map = device_map
        return model
    except Exception as e:
        handle_error(str(e))

# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)


def move_hidden_layers(transformer, h=None):
    if h is None:
        h = transformer.h

    assert len(gpu_blocks) <= torch.cuda.device_count()
    assert sum(gpu_blocks) <= len(h)
    ram_blocks = len(h) - sum(gpu_blocks)

    transformer.extrastorage = {}
    torch.cuda.empty_cache()
    
    able_to_pin_layers = True
    for i in range(ram_blocks):
        h[i].to("cpu")
        transformer.extrastorage[i] = copy.deepcopy(h[i])
        smalltensor = torch.tensor(0).to(primary_device)
        for param1 in h[i].parameters():
            param1.data = smalltensor
        h[i].to(primary_device)
        for param in transformer.extrastorage[i].parameters():
            param.requires_grad = False
            param.data = param.data.detach()
            if able_to_pin_layers:
                try:
                    param.data = param.data.pin_memory()
                except:
                    able_to_pin_layers = False
                    logger.warning(f"You only have enough shared GPU memory for {i} out of {ram_blocks} CPU layers.  Expect suboptimal speed.")
            gc.collect()
            torch.cuda.empty_cache()

    if ram_blocks:
        for param1,param2 in zip(h[0].parameters(),transformer.extrastorage[0].parameters()):
            param1.data = param2.data.to(primary_device, non_blocking=False).detach()

        for param1,param2 in zip(h[ram_blocks-1].parameters(),transformer.extrastorage[ram_blocks-1].parameters()):
            param1.data = param2.data.to(primary_device, non_blocking=False).detach()

    i = ram_blocks
    for j in range(len(gpu_blocks)):
        for _ in range(gpu_blocks[j]):
            h[i].to(j)
            i += 1


def new_forward_neo(
    self,
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    if past_key_values is None:
        past_key_values = (None,) * len(self.h)

    if output_hidden_states:
        all_hidden_states = []
    
    if output_attentions:
        all_attentions = []
    
    if return_dict is None:
        return_dict = self.config.use_return_dict

    if attention_mask is not None:
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_ids.shape, past_key_values)
    
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    if self.config.gradient_checkpointing:
        return_dict = False
        self.gradient_checkpointing = True

    # Process layers
    for i, (layer_module, past_key_value) in enumerate(zip(self.h, past_key_values)):
        layer_outputs = layer_module(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask[i] if head_mask is not None else None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if use_cache:
            past_key_values += (layer_outputs.past_key_values,)

        if output_attentions:
            all_attentions.append(layer_outputs.attentions)

    if not return_dict:
        return (hidden_states, all_hidden_states, all_attentions) if output_hidden_states or output_attentions else hidden_states

    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
    )

def new_forward_xglm(
    self,
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    if past_key_values is None:
        past_key_values = (None,) * len(self.h)

    if output_hidden_states:
        all_hidden_states = []
    
    if output_attentions:
        all_attentions = []
    
    if return_dict is None:
        return_dict = self.config.use_return_dict

    if attention_mask is not None:
        attention_mask = self._prepare_decoder_attention_mask(attention_mask, input_ids.shape, past_key_values)
    
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
    if self.config.gradient_checkpointing:
        return_dict = False
        self.gradient_checkpointing = True

    # Process layers
    for i, (layer_module, past_key_value) in enumerate(zip(self.h, past_key_values)):
        layer_outputs = layer_module(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask[i] if head_mask is not None else None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = layer_outputs[0]

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if use_cache:
            past_key_values += (layer_outputs.past_key_values,)

        if output_attentions:
            all_attentions.append(layer_outputs.attentions)

    if not return_dict:
        return (hidden_states, all_hidden_states, all_attentions) if output_hidden_states or output_attentions else hidden_states

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
    )
