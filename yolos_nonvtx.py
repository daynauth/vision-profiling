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


    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape

        embeddings = self.patch_embeddings(pixel_values)
        
        #torch.cuda.synchronize()

        
        
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
    def __init__(self, config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))

        value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))




        attention_scores = attention_scores / math.sqrt(self.attention_head_size)



        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

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
    def __init__(self, config) -> None:
        super().__init__()
        self.attention = YolosSelfAttention(config)
        self.output = YolosSelfOutput(config)
        self.pruned_heads = set()


    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

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

        hidden_states = self.dense(hidden_states)
        

        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->Yolos
class YolosOutput(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)


        hidden_states = self.dropout(hidden_states)


        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->Yolos
class YolosLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = YolosAttention(config)
        self.intermediate = YolosIntermediate(config)
        self.output = YolosOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        

        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in Yolos, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection

        hidden_states = attention_output + hidden_states

        # in Yolos, layernorm is also applied after self-attention

        layer_output = self.layernorm_after(hidden_states)


        layer_output = self.intermediate(layer_output)

        # second residual connection is done here

        layer_output = self.output(layer_output, hidden_states)

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
        self.layer = nn.ModuleList([YolosLayer(config) for _ in range(config.num_hidden_layers)])

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

        self.interpolation = InterpolateMidPositionEmbeddings(config) if config.use_mid_position_embeddings else None


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

        if self.config.use_mid_position_embeddings:
            interpolated_mid_position_embeddings = self.interpolation(self.mid_position_embeddings, (height, width))


        for i, layer_module in enumerate(self.layer):


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
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
                #torch.cuda.synchronize()


            hidden_states = layer_outputs[0]

            if self.config.use_mid_position_embeddings:
                if i < (self.config.num_hidden_layers - 1):
                    hidden_states = hidden_states + interpolated_mid_position_embeddings[i]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            

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

        embedding_output = self.embeddings(pixel_values)

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


        sequence_output = self.layernorm(sequence_output)
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

        # Initialize weights and apply final processing
        self.post_init()

    # taken from https://github.com/facebookresearch/detr/blob/master/models/detr.py
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


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

        sequence_output = sequence_output[:, -self.config.num_detection_tokens :, :]

        # Class logits + predicted bounding boxes
        logits = self.class_labels_classifier(sequence_output)
        pred_boxes = self.bbox_predictor(sequence_output).sigmoid()

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

def get_mem():
    mem = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"Memory allocated: {mem} MB")

def get_cache():
    cache_size = torch.cuda.memory_reserved() / 1024 / 1024
    print(f"Cache size: {cache_size} MB")

def show_mem(message: str = ""):
    if message != "":
        print(f"--------- START {message}-----------")
    else:
        print("--------------------")

    get_mem()
    get_cache()

    if message != "":
        print(f"--------- END {message}-----------")

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #device = "cpu"

    image = Image.open("000000039769.jpg")
    
    image_processor = AutoImageProcessor.from_pretrained("hustvl/yolos-base")    
    inputs = image_processor(images=image, return_tensors="pt")


    inputs.to(device)


    inputs = inputs['pixel_values']


    model = YolosForObjectDetection.from_pretrained("hustvl/yolos-base").eval().to(device)



    with torch.no_grad():
        model(inputs)

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