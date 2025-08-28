# Contains all the necessary modules needed for the actual computation

import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple
import einops
from einops import rearrange

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

logging.basicConfig(level=logging.INFO)


class FullAttention(nn.Module):
    """
    Full attention using flash attention API.

    Parameters:
        hidden_size: The size of the hidden layers.
        num_heads: The number of attention heads.
        head_dim: The dimension of each attention head.
        use_norm: Whether to use layer normalization.
        norm_eps: The epsilon value for layer normalization.
        layer_idx: The index of the layer.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 16,
        head_dim: int = None,
        norm_eps: float = 1e-5,
        layer_idx: int = None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.layer_idx = layer_idx

        self.norm = nn.LayerNorm(self.hidden_size, eps=norm_eps)

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_attentions: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, q_len, _ = hidden_states.size()

        hidden_states = self.norm(hidden_states)

        q = rearrange(
            self.q_proj(hidden_states), "... (h d) -> ... h d", h=self.num_heads
        )
        k = rearrange(
            self.k_proj(hidden_states), "... (h d) -> ... h d", h=self.num_kv_heads
        )
        v = rearrange(
            self.v_proj(hidden_states), "... (h d) -> ... h d", h=self.num_kv_heads
        )

        if flash_attn_func is None:
            raise ImportError(
                "Please install Flash Attention via `pip install flash-attn --no-build-isolation` first"
            )

        o = flash_attn_func(
            q,
            k,
            v,
            causal=False,  # use non-causal attention for vision
            window_size=(-1, -1),
        )
        o = o.reshape(batch_size, q_len, self.hidden_size)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, None


class Mlp(nn.Module):
    """
    A MLP

    Parameters:
        hidden_size: The size of the hidden layers.
        intermediate_size: The size of the intermediate layers.
    """
    def __init__(self,
        hidden_size: int = 768,
        intermediate_size: int = None,
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 4
        
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class SwishGLU(nn.Module):
    """
    SwishGLU: Gated Linear Unit with Swish activation.
    
    Parameters:
        hidden_size: The size of the input hidden layers.
        intermediate_size: The size of the intermediate layers (before gating).
    """
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = None,
    ):
        super().__init__()
        if intermediate_size is None:
            intermediate_size = hidden_size * 8 / 3
            
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swish = nn.SiLU()
        self.norm = nn.LayerNorm(hidden_size)

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.norm(x)
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        hidden_states = self.swish(gate) * up
        
        return self.down_proj(hidden_states)


class Identity(nn.Module):
    """
    Identity layer.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x