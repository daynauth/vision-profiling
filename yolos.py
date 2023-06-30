import collections
from torch import nn
import torch
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.vit.modeling_vit import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.models.yolos.configuration_yolos import YolosConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.utils.generic import ModelOutput
from PIL import Image

from typing import Optional, Tuple, Union, List, Dict
from dataclasses import dataclass
import math
import time

import nvtx
import sys

fine = True
split = True

num_args = len(sys.argv)

arg1 = "true"
arg2 = "true"

if num_args > 1:
    arg1 = sys.argv[1]

if num_args > 2:
    arg2 = sys.argv[2]

if arg1 == "false":
    fine = False

if arg2 == "false":
    split = False




attention_memory = {
    "layer_0": {
        "mid_position_embedding": 0,
        "query": 30,
        "key": 20,
        "value": 0,
        "mul": 498,
        "div": (1890 - 1370),
        "softmax": (1890 - 1890),
        "dropout": (1890 - 1890),
        "context": (1910 - 1890),
        "Self_Attention_Output": (1910 - 1910)
    },
        "layer_1": {
        "mid_position_embedding": 0,
        "query": (2000 - 2000),
        "key": (2000 - 2000),
        "value": (2000 - 2000),
        "mul": (2520 - 2000),
        "div": (3040 - 2520),
        "softmax": (3040 - 3040),
        "dropout": (3040 - 3040),
        "context": (3040 - 3040),
        "Self_Attention_Output": (3040 - 3040)
    },
        "layer_2": {
        "mid_position_embedding": 0,
        "query": 0,
        "key": 0,
        "value": 0,
        "mul": 520,
        "div": 520,
        "softmax": 0,
        "dropout": 0,
        "context": 0,
        "Self_Attention_Output": 0
    },
        "layer_3": {
        "mid_position_embedding": 0,
        "query": 0,
        "key": 0,
        "value": 0,
        "mul": 520,
        "div": 520,
        "softmax": 0,
        "dropout": 0,
        "context": 0,
        "Self_Attention_Output": 0
    },
        "layer_4": {
        "mid_position_embedding": 0,
        "query": 0,
        "key": 0,
        "value": 0,
        "mul": 520,
        "div": 520,
        "softmax": 0,
        "dropout": 0,
        "context": 0,
        "Self_Attention_Output": 0
    },
        "layer_5": {
        "mid_position_embedding": 0,
        "query": 0,
        "key": 0,
        "value": 0,
        "mul": 520,
        "div": 520,
        "softmax": 0,
        "dropout": 0,
        "context": 0,
        "Self_Attention_Output": 0
    },
        "layer_6": {
        "mid_position_embedding": 0,
        "query": 0,
        "key": 0,
        "value": 0,
        "mul": 520,
        "div": 520,
        "softmax": 0,
        "dropout": 0,
        "context": 0,
        "Self_Attention_Output": 0
    },
        "layer_7": {
            "mid_position_embedding": 0,
            "query": 0,
            "key": 0,
            "value": 0,
            "mul": 520,
            "div": 520,
            "softmax": 0,
            "dropout": 0,
            "context": 0,
            "Self_Attention_Output": 0
    },
        "layer_8": {
            "mid_position_embedding": 0,
            "query": 0,
            "key": 0,
            "value": 0,
            "mul": 520,
            "div": 520,
            "softmax": 0,
            "dropout": 0,
            "context": 0,
            "Self_Attention_Output": 0
        },
        "layer_9": {
            "mid_position_embedding": 0,
            "query": 0,
            "key": 0,
            "value": 0,
            "mul": 520,
            "div": 520,
            "softmax": 0,
            "dropout": 0,
            "context": 0,
            "Self_Attention_Output": 0
        },
        "layer_10": {
            "mid_position_embedding": 0,
            "query": 0,
            "key": 0,
            "value": 0,
            "mul": 520,
            "div": 520,
            "softmax": 0,
            "dropout": 0,
            "context": 0,
            "Self_Attention_Output": 0
        },
        "layer_11": {
            "mid_position_embedding": 0,
            "query": 0,
            "key": 0,
            "value": 0,
            "mul": 520,
            "div": 520,
            "softmax": 0,
            "dropout": 0,
            "context": 0,
            "Self_Attention_Output": 0
        },
}


class YolosEmbeddings(nn.Module):
    """
    Construct the CLS token, detection tokens, position and patch embeddings.

    """

    def __init__(self, config, logging = False) -> None:
        super().__init__()
        self.logging = logging
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.detection_tokens = nn.Parameter(torch.zeros(1, config.num_detection_tokens, config.hidden_size))
        self.patch_embeddings = YolosPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches + config.num_detection_tokens + 1, config.hidden_size)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.interpolation = InterpolateInitialPositionEmbeddings(config)
        self.config = config

    @nvtx.annotate("YolosEmbeddings.forward", color="blue")
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape

        embeddings = self.patch_embeddings(pixel_values)
        
        torch.cuda.synchronize()

        
        
        batch_size, seq_len, _ = embeddings.size()

        # add the [CLS] and detection tokens to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        detection_tokens = self.detection_tokens.expand(batch_size, -1, -1)


        embeddings = torch.cat((cls_tokens, embeddings, detection_tokens), dim=1)




        # add positional encoding to each token
        # this might require interpolation of the existing position embeddings
        position_embeddings = self.interpolation(self.position_embeddings, (height, width))


        embeddings = embeddings + position_embeddings

        embeddings = self.dropout(embeddings)


        return embeddings


class InterpolateInitialPositionEmbeddings(nn.Module):
    def __init__(self, config, logging = False) -> None:
        super().__init__()
        self.logging = logging
        self.config = config

    def forward(self, pos_embed, img_size=(800, 1344)) -> torch.Tensor:
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, -self.config.num_detection_tokens :, :]
        patch_pos_embed = pos_embed[:, 1 : -self.config.num_detection_tokens, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)

        batch_size, hidden_size, seq_len = patch_pos_embed.shape

        patch_height, patch_width = (
            self.config.image_size[0] // self.config.patch_size,
            self.config.image_size[1] // self.config.patch_size,
        )
        patch_pos_embed = patch_pos_embed.view(batch_size, hidden_size, patch_height, patch_width)

        height, width = img_size
        new_patch_heigth, new_patch_width = height // self.config.patch_size, width // self.config.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_patch_heigth, new_patch_width), mode="bicubic", align_corners=False
        )
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)

        return scale_pos_embed


class YolosPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, logging = False):
        super().__init__()
        self.logging = logging
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)


        return embeddings
    

# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->Yolos
class YolosSelfAttention(nn.Module):
    def __init__(self, config, layer_name) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )
        self.layer_name = layer_name
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)


    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        with nvtx.annotate(f"{self.layer_name}_Query", color="blue"):
            self.start.record()
            query_layer = self.transpose_for_scores(self.query(hidden_states))
            self.end.record()
            torch.cuda.synchronize()
            end_time = self.start.elapsed_time(self.end)
            query_layer_size = query_layer.element_size() * query_layer.nelement() / 1024 / 1024
            query_layer_weight_size = self.query.weight.element_size() * self.query.weight.nelement() / 1024 / 1024
            query_layer_bias_size = self.query.bias.element_size() * self.query.bias.nelement() / 1024 / 1024
            query_layer_param_size = query_layer_weight_size + query_layer_bias_size
            if fine:
                print(f"{self.layer_name}_Query, {end_time/1000},0, {attention_memory[self.layer_name]['query'] + query_layer_param_size},{query_layer_size},0")

        with nvtx.annotate(f"{self.layer_name}_Key", color="purple"):
            self.start.record()
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            self.end.record()
            torch.cuda.synchronize()
            end_time = self.start.elapsed_time(self.end)
            key_layer_size = key_layer.element_size() * key_layer.nelement() / 1024 / 1024
            key_layer_weight_size = self.key.weight.element_size() * self.key.weight.nelement() / 1024 / 1024
            key_layer_bias_size = self.key.bias.element_size() * self.key.bias.nelement() / 1024 / 1024
            key_layer_param_size = key_layer_weight_size + key_layer_bias_size
            if fine:
                print(f"{self.layer_name}_Key, {end_time/1000},0,{attention_memory[self.layer_name]['key'] + key_layer_param_size},{key_layer_size},0")

        with nvtx.annotate(f"{self.layer_name}_Value", color="green"):
            self.start.record()
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            self.end.record()
            torch.cuda.synchronize()
            end_time = self.start.elapsed_time(self.end)
            value_layer_size = value_layer.element_size() * value_layer.nelement() / 1024 / 1024
            value_layer_weight_size = self.value.weight.element_size() * self.value.weight.nelement() / 1024 / 1024
            value_layer_bias_size = self.value.bias.element_size() * self.value.bias.nelement() / 1024 / 1024
            value_layer_param_size = value_layer_weight_size + value_layer_bias_size
            if fine:
                print(f"{self.layer_name}_Value, {end_time/1000},0,{attention_memory[self.layer_name]['value'] + value_layer_param_size},{value_layer_size},0")


        # Take the dot product between "query" and "key" to get the raw attention scores.
        with nvtx.annotate(f"{self.layer_name}_mul", color="red"):
            self.start.record()
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            self.end.record()
            torch.cuda.synchronize()
            end_time = self.start.elapsed_time(self.end)
            attention_scores_size = attention_scores.element_size() * attention_scores.nelement() / 1024 / 1024
            if fine:
                print(f"{self.layer_name}_mul, {end_time/1000},0,{attention_memory[self.layer_name]['mul']},{attention_scores_size},0")



        with nvtx.annotate(f"{self.layer_name}_div", color="red"):
            self.start.record()
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            self.end.record()
            torch.cuda.synchronize()
            end_time = self.start.elapsed_time(self.end)
            attention_scores_size = attention_scores.element_size() * attention_scores.nelement() / 1024 / 1024
            if fine:
                print(f"{self.layer_name}_div, {end_time/1000},0,{attention_memory[self.layer_name]['div']},{attention_scores_size},0")


        # Normalize the attention scores to probabilities.
        with nvtx.annotate(f"{self.layer_name}_softmax", color="red"):
            self.start.record()
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            self.end.record()
            torch.cuda.synchronize()
            end_time = self.start.elapsed_time(self.end)
            attention_probs_size = attention_probs.element_size() * attention_probs.nelement() / 1024 / 1024
            if fine:
                print(f"{self.layer_name}_softmax, {end_time/1000},0,{attention_memory[self.layer_name]['softmax']},{attention_probs_size},0")

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with nvtx.annotate(f"{self.layer_name}_dropout", color="red"):
            self.start.record()
            attention_probs = self.dropout(attention_probs)
            self.end.record()
            torch.cuda.synchronize()
            end_time = self.start.elapsed_time(self.end)
            attention_probs_size = attention_probs.element_size() * attention_probs.nelement() / 1024 / 1024
            if fine:
                print(f"{self.layer_name}_dropout, {end_time/1000},0,{attention_memory[self.layer_name]['dropout']},{attention_probs_size},0")

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        with nvtx.annotate(f"{self.layer_name}_context", color="red"):
            self.start.record()
            context_layer = torch.matmul(attention_probs, value_layer)

            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
            self.end.record()
            torch.cuda.synchronize()
            end_time = self.start.elapsed_time(self.end)
            context_layer_size = context_layer.element_size() * context_layer.nelement() / 1024 / 1024
            if fine:
                print(f"{self.layer_name}_context, {end_time/1000},0,{attention_memory[self.layer_name]['context']},{context_layer_size},0")

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->Yolos
class YolosSelfOutput(nn.Module):
    """
    The residual connection is defined in YolosLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
 
# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->Yolos
class YolosAttention(nn.Module):
    def __init__(self, config, layer_name) -> None:
        super().__init__()
        self.attention = YolosSelfAttention(config, layer_name)
        self.output = YolosSelfOutput(config)
        self.pruned_heads = set()

        self.layer_name = layer_name
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)


    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        with nvtx.annotate(f"{self.layer_name}_Self_Attention_Output", color="purple"):
            self.start.record()
            attention_output = self.output(self_outputs[0], hidden_states)
            self.end.record()
            torch.cuda.synchronize()
            end_time = self.start.elapsed_time(self.end)
            attention_output_size = attention_output.element_size() * attention_output.nelement() / 1024 / 1024
            attention_output_weight_size = self.output.dense.weight.element_size() * self.output.dense.weight.nelement() / 1024 / 1024
            attention_output_bias_size = self.output.dense.bias.element_size() * self.output.dense.bias.nelement() / 1024 / 1024
            attention_output_param_size = attention_output_weight_size + attention_output_bias_size
            if fine:
                print(f"{self.layer_name}_Self_Attention_Output, {end_time/1000},0,{attention_memory[self.layer_name]['Self_Attention_Output'] + attention_output_param_size},{attention_output_size},0")

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->Yolos
class YolosIntermediate(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        with nvtx.annotate("Intermediate Dense", color="blue"):
            hidden_states = self.dense(hidden_states)
        
        #print("Intermediate Dense Output: ", hidden_states.shape)
        #print("Size of Dense Output in MB: ", hidden_states.element_size() * hidden_states.nelement() / 1024 / 1024)
        with nvtx.annotate("Intermediate Activation", color="purple"):
            hidden_states = self.intermediate_act_fn(hidden_states)

        #print("Intermediate Activation Output: ", hidden_states.shape)
        #print("Size of Activation Output in MB: ", hidden_states.element_size() * hidden_states.nelement() / 1024 / 1024)
        return hidden_states

# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->Yolos
class YolosOutput(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        with nvtx.annotate("Output Dense", color="blue"):
            hidden_states = self.dense(hidden_states)

        with nvtx.annotate("Output Dropout", color="purple"):
            hidden_states = self.dropout(hidden_states)

        with nvtx.annotate("Output Residual Connection", color="green"):
            hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->Yolos
class YolosLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config, layer_index:str ) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        
        self.intermediate = YolosIntermediate(config)
        self.output = YolosOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.layer_name = "layer_" + layer_index

        self.attention = YolosAttention(config, self.layer_name)

    @nvtx.annotate("Attention Head Calculation", color="red")
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        with nvtx.annotate(f"{self.layer_name}_Layer_Norm_Before", color="purple"):
            self.starter.record()
            norm_output = self.layernorm_before(hidden_states)# in Yolos, layernorm is applied before self-attention
            self.ender.record()
            torch.cuda.synchronize()
            end_time = self.starter.elapsed_time(self.ender)
            norm_output_size = norm_output.element_size() * norm_output.nelement() / 1024 / 1024
            norm_output_weight_size = self.layernorm_before.weight.element_size() * self.layernorm_before.weight.nelement() / 1024 / 1024
            norm_output_bias_size = self.layernorm_before.bias.element_size() * self.layernorm_before.bias.nelement() / 1024 / 1024
            norm_param_size = norm_output_size + norm_output_weight_size + norm_output_bias_size
            if fine:
                print(f"{self.layer_name}_Layer_Norm_Before, {end_time/1000},0,{norm_param_size},{norm_output_size},0")

        with nvtx.annotate(f"{self.layer_name}_Self_Attention_Forward", color="purple"):
            self.starter.record()
            self_attention_outputs = self.attention(
                norm_output,  
                head_mask,
                output_attentions=output_attentions,
            )
            self.ender.record()
            torch.cuda.synchronize()
            end_time = self.starter.elapsed_time(self.ender)
            attention_output_size = self_attention_outputs[0].element_size() * self_attention_outputs[0].nelement() / 1024 / 1024
            #print(f"{self.layer_name}_Self_Attention_Forward, {end_time/1000},0,1140,{attention_output_size},0")

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        with nvtx.annotate(f"{self.layer_name}_Residual_Connection_1", color="red"):
            self.starter.record()
            hidden_states = attention_output + hidden_states
            self.ender.record()
            torch.cuda.synchronize()
            end_time = self.starter.elapsed_time(self.ender)
            hidden_states_size = hidden_states.element_size() * hidden_states.nelement() / 1024 / 1024
            if fine:
                print(f"{self.layer_name}_Residual_Connection_1, {end_time/1000},0,0,{hidden_states_size},0")

        # in Yolos, layernorm is also applied after self-attention
        with nvtx.annotate(f"{self.layer_name}_Layer_Norm_After", color="purple"):
            self.starter.record()
            layer_output = self.layernorm_after(hidden_states)
            self.ender.record()
            torch.cuda.synchronize()
            end_time = self.starter.elapsed_time(self.ender)
            layer_output_size = layer_output.element_size() * layer_output.nelement() / 1024 / 1024
            layer_output_weight_size = self.layernorm_after.weight.element_size() * self.layernorm_after.weight.nelement() / 1024 / 1024
            layer_output_bias_size = self.layernorm_after.bias.element_size() * self.layernorm_after.bias.nelement() / 1024 / 1024
            layer_output_param_size = layer_output_weight_size + layer_output_bias_size
            if fine:
                print(f"{self.layer_name}_Layer_Norm_After, {end_time/1000},0,{20 + layer_output_param_size},{layer_output_size},0")

        with nvtx.annotate(f"{self.layer_name}_Intermediate_Forward", color="green"):
            self.starter.record()
            layer_output = self.intermediate(layer_output)
            self.ender.record()
            torch.cuda.synchronize()
            end_time = self.starter.elapsed_time(self.ender)
            layer_output_size = layer_output.element_size() * layer_output.nelement() / 1024 / 1024
            layer_output_weight_size = self.intermediate.dense.weight.element_size() * self.intermediate.dense.weight.nelement() / 1024 / 1024
            layer_output_bias_size = self.intermediate.dense.bias.element_size() * self.intermediate.dense.bias.nelement() / 1024 / 1024
            layer_output_param_size = layer_output_weight_size + layer_output_bias_size
            if fine:
                print(f"{self.layer_name}_Intermediate_Forward, {end_time/1000},0,{70 + layer_output_param_size},{layer_output_size},0")

        # second residual connection is done here
        with nvtx.annotate(f"{self.layer_name}_Output", color="red"):
            self.starter.record()
            layer_output = self.output(layer_output, hidden_states)
            self.ender.record()
            torch.cuda.synchronize()
            end_time = self.starter.elapsed_time(self.ender)
            layer_output_size = layer_output.element_size() * layer_output.nelement() / 1024 / 1024
            layer_output_weight_size = self.output.dense.weight.element_size() * self.output.dense.weight.nelement() / 1024 / 1024
            layer_output_bias_size = self.output.dense.bias.element_size() * self.output.dense.bias.nelement() / 1024 / 1024
            layer_output_param_size = layer_output_weight_size + layer_output_bias_size
            if fine:
                print(f"{self.layer_name}_Output, {end_time/1000},0,{layer_output_param_size},{layer_output_size},0")

        outputs = (layer_output,) + outputs

        return outputs


class InterpolateMidPositionEmbeddings(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, pos_embed, img_size=(800, 1344)) -> torch.Tensor:
        cls_pos_embed = pos_embed[:, :, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, :, -self.config.num_detection_tokens :, :]
        patch_pos_embed = pos_embed[:, :, 1 : -self.config.num_detection_tokens, :]
        patch_pos_embed = patch_pos_embed.transpose(2, 3)
        depth, batch_size, hidden_size, seq_len = patch_pos_embed.shape

        patch_height, patch_width = (
            self.config.image_size[0] // self.config.patch_size,
            self.config.image_size[1] // self.config.patch_size,
        )
        patch_pos_embed = patch_pos_embed.view(depth * batch_size, hidden_size, patch_height, patch_width)
        height, width = img_size
        new_patch_height, new_patch_width = height // self.config.patch_size, width // self.config.patch_size
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed, size=(new_patch_height, new_patch_width), mode="bicubic", align_corners=False
        )
        patch_pos_embed = (
            patch_pos_embed.flatten(2)
            .transpose(1, 2)
            .contiguous()
            .view(depth, batch_size, new_patch_height * new_patch_width, hidden_size)
        )
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=2)
        return scale_pos_embed
    
class YolosEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.split = split
        self.layer = nn.ModuleList([YolosLayer(config, str(i)) for i in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False

        seq_length = (
            1 + (config.image_size[0] * config.image_size[1] // config.patch_size**2) + config.num_detection_tokens
        )
        self.mid_position_embeddings = (
            nn.Parameter(
                torch.zeros(
                    config.num_hidden_layers - 1,
                    1,
                    seq_length,
                    config.hidden_size,
                )
            )
            if config.use_mid_position_embeddings
            else None
        )

        print(self.mid_position_embeddings.shape)

        self.interpolation = InterpolateMidPositionEmbeddings(config) if config.use_mid_position_embeddings else None

        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    @nvtx.annotate("YolosEncoder.forward", color="green")
    def forward(
        self,
        hidden_states: torch.Tensor,
        height,
        width,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        self.starter.record()
        with nvtx.annotate("Mid Position Embedding", color="purple"):
            if self.config.use_mid_position_embeddings and not self.split:
                interpolated_mid_position_embeddings = self.interpolation(self.mid_position_embeddings, (height, width))

                self.ender.record()
                torch.cuda.synchronize()
                end_time = self.starter.elapsed_time(self.ender)
                interpolated_mid_position_embeddings_size = interpolated_mid_position_embeddings.element_size() * interpolated_mid_position_embeddings.nelement() / 1024 / 1024
                interpolated_mid_position_embeddings_weight_size = self.mid_position_embeddings.element_size() * self.mid_position_embeddings.nelement() / 1024 / 1024
                print(f"Interpolation, {end_time/1000},0,{218 + interpolated_mid_position_embeddings_weight_size},{interpolated_mid_position_embeddings_size},0")
        memory = [
            1782,
            1120,
            1130,
            1140,
            1140,
            1130,
            1190,
            1130,
            1180,
            1110,
            1170,
            1110
        ]
        for i, layer_module in enumerate(self.layer):

            rng = nvtx.start_range(message=f"encoder_{i}", color="blue")
            self.starter.record()

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                self.starter.record()
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
                self.ender.record()
                torch.cuda.synchronize()
                #print(f"layer_{i},{self.starter.elapsed_time(self.ender)/1000},0,{memory[i]},{layer_outputs[0].element_size() * layer_outputs[0].nelement() / 1024 / 1024},0")
                nvtx.end_range(rng)

            hidden_states = layer_outputs[0]
           
            if self.config.use_mid_position_embeddings:
                if i < (self.config.num_hidden_layers - 1):
                    if self.split:
                        with nvtx.annotate(f"layer_{i}_mid_position_embedding", color="purple"):
                            self.starter.record()
                            interpolated_mid_position_embedding = self.interpolation(self.mid_position_embeddings[i].unsqueeze(0), (height, width))
                            self.ender.record()
                            torch.cuda.synchronize()
                            end_time = self.starter.elapsed_time(self.ender)
                            interpolated_mid_position_embedding_size = interpolated_mid_position_embedding.element_size() * interpolated_mid_position_embedding.nelement() / 1024 / 1024
                            interpolated_mid_position_embedding_weight_size = self.mid_position_embeddings[i].element_size() * self.mid_position_embeddings[i].nelement() / 1024 / 1024
                            layer = f"layer_{i}"
                            print(f"layer_{i}_mid_position_embedding, {end_time/1000},0,{attention_memory[layer]['mid_position_embedding'] + interpolated_mid_position_embedding_weight_size/11},{interpolated_mid_position_embedding_size},0")


                        with nvtx.annotate(f"layer_{i}_add_mid_position_embedding", color="purple"):
                            self.starter.record()
                            hidden_states = hidden_states + interpolated_mid_position_embedding[0]
                            self.ender.record()
                            torch.cuda.synchronize()
                            end_time = self.starter.elapsed_time(self.ender)
                            hidden_states_size = hidden_states.element_size() * hidden_states.nelement() / 1024 / 1024
                            print(f"layer_{i}_add_mid_position_embedding, {end_time/1000},0,0,{hidden_states_size},0")
                    else:
                        with nvtx.annotate(f"layer_{i}_add_mid_position_embedding", color="purple"):
                            self.starter.record()
                            hidden_states = hidden_states + interpolated_mid_position_embeddings[i]
                            self.ender.record()
                            torch.cuda.synchronize()
                            end_time = self.starter.elapsed_time(self.ender)
                            hidden_states_size = hidden_states.element_size() * hidden_states.nelement() / 1024 / 1024
                            print(f"layer_{i}_add_mid_position_embedding, {end_time/1000},0,0,{hidden_states_size},0")

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
            
            self.ender.record()
            torch.cuda.synchronize()
            end_time = self.starter.elapsed_time(self.ender)

            if not fine:
                layer_output_size = hidden_states.element_size() * hidden_states.nelement() / 1024 / 1024
                query_weight_size = layer_module.attention.attention.query.weight.element_size() * layer_module.attention.attention.query.weight.nelement() / 1024 / 1024
                query_bias_size = layer_module.attention.attention.query.bias.element_size() * layer_module.attention.attention.query.bias.nelement() / 1024 / 1024
                key_weight_size = layer_module.attention.attention.key.weight.element_size() * layer_module.attention.attention.key.weight.nelement() / 1024 / 1024
                key_bias_size = layer_module.attention.attention.key.bias.element_size() * layer_module.attention.attention.key.bias.nelement() / 1024 / 1024
                value_weight_size = layer_module.attention.attention.value.weight.element_size() * layer_module.attention.attention.value.weight.nelement() / 1024 / 1024
                value_bias_size = layer_module.attention.attention.value.bias.element_size() * layer_module.attention.attention.value.bias.nelement() / 1024 / 1024
                output_dense_weight_size = layer_module.attention.output.dense.weight.element_size() * layer_module.attention.output.dense.weight.nelement() / 1024 / 1024
                output_dense_bias_size = layer_module.attention.output.dense.bias.element_size() * layer_module.attention.output.dense.bias.nelement() / 1024 / 1024
                attention_param_size = query_weight_size + query_bias_size + key_weight_size + key_bias_size + value_weight_size + value_bias_size + output_dense_weight_size + output_dense_bias_size

                intermediate_dense_weight_size = layer_module.intermediate.dense.weight.element_size() * layer_module.intermediate.dense.weight.nelement() / 1024 / 1024
                intermediate_dense_bias_size = layer_module.intermediate.dense.bias.element_size() * layer_module.intermediate.dense.bias.nelement() / 1024 / 1024
                intermediate_param_size = intermediate_dense_weight_size + intermediate_dense_bias_size

                output_dense_weight_size = layer_module.output.dense.weight.element_size() * layer_module.output.dense.weight.nelement() / 1024 / 1024
                output_dense_bias_size = layer_module.output.dense.bias.element_size() * layer_module.output.dense.bias.nelement() / 1024 / 1024
                output_param_size = output_dense_weight_size + output_dense_bias_size

                layer_norm_weight_size = layer_module.layernorm_before.weight.element_size() * layer_module.layernorm_before.weight.nelement() / 1024 / 1024
                layer_norm_bias_size = layer_module.layernorm_before.bias.element_size() * layer_module.layernorm_before.bias.nelement() / 1024 / 1024
                layer_norm_param_size = layer_norm_weight_size + layer_norm_bias_size

                layer_norm_after_weight_size = layer_module.layernorm_after.weight.element_size() * layer_module.layernorm_after.weight.nelement() / 1024 / 1024
                layer_norm_after_bias_size = layer_module.layernorm_after.bias.element_size() * layer_module.layernorm_after.bias.nelement() / 1024 / 1024
                layer_norm_after_param_size = layer_norm_after_weight_size + layer_norm_after_bias_size

                memory[i] = attention_param_size + intermediate_param_size + output_param_size + layer_norm_param_size + layer_norm_after_param_size

                print(f"layer_{i}, {end_time/1000},0,{memory[i]},{layer_output_size},0")
            

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class YolosPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = YolosConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module: YolosEncoder, value: bool = False) -> None:
        if isinstance(module, YolosEncoder):
            module.gradient_checkpointing = value


class YolosModel(YolosPreTrainedModel):
    def __init__(self, config, add_pooling_layer: bool = True):
        super().__init__(config)
        self.config = config

        self.embeddings = YolosEmbeddings(config)
        self.encoder = YolosEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> YolosPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model.

        Args:
            heads_to_prune (`dict` of {layer_num: list of heads to prune in this layer}):
                See base class `PreTrainedModel`.
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        self.starter.record()
        embedding_output = self.embeddings(pixel_values)
        self.ender.record()
        torch.cuda.synchronize()
        end_time = self.starter.elapsed_time(self.ender)
        embedding_output_size = embedding_output.element_size() * embedding_output.nelement() / 1024 / 1024

        embedding_output_weight_size = self.embeddings.patch_embeddings.projection.weight.element_size() * self.embeddings.patch_embeddings.projection.weight.nelement() / 1024 / 1024
        embedding_output_bias_size = self.embeddings.patch_embeddings.projection.bias.element_size() * self.embeddings.patch_embeddings.projection.bias.nelement() / 1024 / 1024

        embedding_cls_weight_size = self.embeddings.cls_token.element_size() * self.embeddings.cls_token.nelement() / 1024 / 1024
        embedding_det_weight_size = self.embeddings.detection_tokens.element_size() * self.embeddings.detection_tokens.nelement() / 1024 / 1024

        embedding_pos_weight_size = self.embeddings.position_embeddings.element_size() * self.embeddings.position_embeddings.nelement() / 1024 / 1024

        embedding_param_size = embedding_output_weight_size + embedding_output_bias_size + embedding_cls_weight_size + embedding_det_weight_size + embedding_pos_weight_size


        print(f"Embedding, {end_time/1000},0,{48.14 + embedding_param_size},{embedding_output_size},0")


        encoder_outputs = self.encoder(
            embedding_output,
            height=pixel_values.shape[-2],
            width=pixel_values.shape[-1],
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]

        with nvtx.annotate("YolosModel.forward.layernorm", color="purple"):
            self.starter.record()
            sequence_output = self.layernorm(sequence_output)
            self.ender.record()
            torch.cuda.synchronize()
            end_time = self.starter.elapsed_time(self.ender)
            layer_norm_size = sequence_output.element_size() * sequence_output.nelement() / 1024 / 1024
            print(f"Layer_Norm, {end_time/1000},0,0,{layer_norm_size},0")
        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=None,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead with Detr->Yolos
class YolosMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    @nvtx.annotate("YolosMLPPredictionHead.forward", color="black")
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@dataclass
class YolosObjectDetectionOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class YolosForObjectDetection(YolosPreTrainedModel):
    def __init__(self, config: YolosConfig):
        super().__init__(config)

     

        # YOLOS (ViT) encoder model
        self.vit = YolosModel(config, add_pooling_layer=False)

        

        # Object detection heads
        # We add one for the "no object" class
        self.class_labels_classifier = YolosMLPPredictionHead(
            input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=config.num_labels + 1, num_layers=3
        )
        self.bbox_predictor = YolosMLPPredictionHead(
            input_dim=config.hidden_size, hidden_dim=config.hidden_size, output_dim=4, num_layers=3
        )

        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    @nvtx.annotate("YolosForObjectDetection.forward", color="red")
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[List[Dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, YolosObjectDetectionOutput]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        # First, sent images through YOLOS base model to obtain hidden states
        outputs = self.vit(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = outputs[0]

        # Take the final hidden states of the detection tokens
        with nvtx.annotate("sequence reshape", color="red"):
            sequence_output = sequence_output[:, -self.config.num_detection_tokens :, :]

        # Class logits + predicted bounding boxes
        self.starter.record()
        logits = self.class_labels_classifier(sequence_output)
        self.ender.record()
        torch.cuda.synchronize()
        end_time = self.starter.elapsed_time(self.ender)
        logits_size = logits.element_size() * logits.nelement() / 1024 / 1024
        logits_weight_size = self.class_labels_classifier.layers[0].weight.element_size() * self.class_labels_classifier.layers[0].weight.nelement() / 1024 / 1024 + \
            self.class_labels_classifier.layers[1].weight.element_size() * self.class_labels_classifier.layers[1].weight.nelement() / 1024 / 1024 + \
            self.class_labels_classifier.layers[2].weight.element_size() * self.class_labels_classifier.layers[2].weight.nelement() / 1024 / 1024
        logits_bias_size = self.class_labels_classifier.layers[0].bias.element_size() * self.class_labels_classifier.layers[0].bias.nelement() / 1024 / 1024 + \
            self.class_labels_classifier.layers[1].bias.element_size() * self.class_labels_classifier.layers[1].bias.nelement() / 1024 / 1024 + \
            self.class_labels_classifier.layers[2].bias.element_size() * self.class_labels_classifier.layers[2].bias.nelement() / 1024 / 1024
        logits_param_size = logits_weight_size + logits_bias_size
        print(f"Class_Labels_Classifier, {end_time/1000},0,{logits_param_size},{logits_size},0")

        self.starter.record()
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()
        self.ender.record()
        torch.cuda.synchronize()
        end_time = self.starter.elapsed_time(self.ender)
        pred_boxes_size = pred_boxes.element_size() * pred_boxes.nelement() / 1024 / 1024
        pred_boxes_weight_size = self.bbox_predictor.layers[0].weight.element_size() * self.bbox_predictor.layers[0].weight.nelement() / 1024 / 1024 + \
            self.bbox_predictor.layers[1].weight.element_size() * self.bbox_predictor.layers[1].weight.nelement() / 1024 / 1024 + \
            self.bbox_predictor.layers[2].weight.element_size() * self.bbox_predictor.layers[2].weight.nelement() / 1024 / 1024
        pred_boxes_bias_size = self.bbox_predictor.layers[0].bias.element_size() * self.bbox_predictor.layers[0].bias.nelement() / 1024 / 1024 + \
            self.bbox_predictor.layers[1].bias.element_size() * self.bbox_predictor.layers[1].bias.nelement() / 1024 / 1024 + \
            self.bbox_predictor.layers[2].bias.element_size() * self.bbox_predictor.layers[2].bias.nelement() / 1024 / 1024
        
        pred_boxes_param_size = pred_boxes_weight_size + pred_boxes_bias_size
        print(f"Box_Predictor, {end_time/1000},0,{pred_boxes_param_size},{pred_boxes_size},0")
        
        loss, loss_dict, auxiliary_outputs = None, None, None


        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes) + outputs
            return ((loss, loss_dict) + output) if loss is not None else output

        return YolosObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    image = Image.open("000000039769.jpg")
    
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-base")    
    inputs = image_processor(images=image, return_tensors="pt")

    with nvtx.annotate("Image load", color="red"):
        inputs.to(device)


    inputs = inputs['pixel_values']
    input_size = inputs.element_size() * inputs.nelement() / 1024 / 1024
    print(f"Input size: {input_size} MB")

    with nvtx.annotate("Model load", color="green"):
        model = YolosForObjectDetection.from_pretrained("hustvl/yolos-base").eval().to(device)



    with torch.no_grad():
        model(inputs)

        start_time = time.time()
        outputs = model(inputs)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time}")

    # target_sizes = torch.tensor([image.size[::-1]])
    # results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
    #     0
    # ]



    # for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    #     box = [round(i, 2) for i in box.tolist()]
    #     print(
    #         f"Detected {model.config.id2label[label.item()]} with confidence "
    #         f"{round(score.item(), 3)} at location {box}"
    #     )

    #print(model)





test()
