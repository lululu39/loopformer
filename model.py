# The actual model

import torch
from torch import nn
from .modules import *
from .router import Router
from .utils import ImageEmbeddings

class LoopFormer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_loops = config.num_loops
        
        self.module_pool = nn.ModuleList()
        
        # Add modules to pool based on config
        for module_type, count in config.modules.items():
            for _ in range(count):
                if module_type == "full_attention":
                    module = FullAttention(
                        hidden_size=self.hidden_size,
                        num_heads=config.num_heads,
                        head_dim=config.head_dim
                    )
                elif module_type == "mlp":
                    module = Mlp(
                        hidden_size=self.hidden_size,
                        intermediate_size=config.intermediate_size
                    )
                elif module_type == "swish_glu":
                    module = SwishGLU(
                        hidden_size=self.hidden_size,
                        intermediate_size=config.intermediate_size
                    )
                elif module_type == "identity":
                    module = Identity()
                else:
                    raise ValueError(f"Unknown module type: {module_type}")
                
                self.module_pool.append(module)
        
        # Router for each turn
        self.routers = nn.ModuleList([
            Router(
                hidden_size=self.hidden_size,
                num_choice=len(self.module_pool)
            ) for _ in range(self.num_loops)
        ])
        
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with router-based module selection
        
        Args:
            hidden_states: Input tensor [B, L, D]
            
        Returns:
            Output tensor [B, L, D]
        """
        for loop in range(self.num_loops):
            route_probs = self.routers[loop](hidden_states)  # [num_choices]
            idx = torch.argmax(route_probs)  # scalar
            
            chosen_module = self.module_pool[idx]
            output = chosen_module(hidden_states)
            
            # Residual and update
            hidden_states = hidden_states + output

        return hidden_states
        
