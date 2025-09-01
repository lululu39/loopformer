import collections.abc
import torch
from torch import nn
from typing import Optional
from modules import *
from mixer import Mixer

class PatchEmbeddings(nn.Module):
    """
    Convert image into patch embeddings.
    Adapted from huggingface/transformers ViT implementation.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = (
            image_size
            if isinstance(image_size, collections.abc.Iterable)
            else (image_size, image_size)
        )
        patch_size = (
            patch_size
            if isinstance(patch_size, collections.abc.Iterable)
            else (patch_size, patch_size)
        )
        num_patches = (image_size[1] // patch_size[1]) * (
            image_size[0] // patch_size[0]
        )
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ImageEmbeddings(nn.Module):
    """
    Construct the position and patch embeddings.
    Adapted from huggingface/transformers ViT implementation. No cls token is used in this implementation.
    """

    def __init__(self, config, use_mask_token: bool = False) -> None:
        super().__init__()

        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            if use_mask_token
            else None
        )
        self.patch_embeddings = PatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches, config.hidden_size)
        )
        self.dropout = nn.Dropout(0.0)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(
        self, embeddings: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1]

        if (
            not torch.jit.is_tracing()
            and num_patches == num_positions
            and height == width
        ):
            return self.position_embeddings

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)
        pos_embed = self.position_embeddings.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )

        pos_embed = pos_embed.permute(0, 3, 1, 2)

        pos_embed = nn.functional.interpolate(
            pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return pos_embed

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings

class Pooler(nn.Module):
    """
    Pool the output of a vision model by taking the mean of all tokens.
    Adapted from huggingface/transformers ViT implementation.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = hidden_states.mean(dim=1)  # always use mean pooling
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

def get_module(type, config):
    if type == "full_attention":
        return FullAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            norm_eps=config.layer_norm_eps
        )
    elif type == "mlp":
        return Mlp(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            norm_eps=config.layer_norm_eps
        )
    elif type == "swish_glu":
        return SwishGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            norm_eps=config.layer_norm_eps
        )
    elif type == "identity":
        return Identity()
    elif "mixer" in type:
        return get_mixer(type, config)
    else:
        raise ValueError(f"Unknown module type: {type}")


def get_mixer(type, config):
    if type == "mixer_1":
        return Mixer(
            token_mixer=get_module("full_attention", config),
            channel_mixer=get_module("mlp", config),
        )
    elif type == "mixer_2":
        return Mixer(
            token_mixer=get_module("full_attention", config),
            channel_mixer=get_module("swish_glu", config),
        )
    else:
        raise ValueError(f"Unknown mixer type: {type}")