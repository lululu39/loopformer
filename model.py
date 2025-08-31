# The actual model

import torch
from torch import nn
from modules import *
from router import *
from utils import *
from config import *
from transformers.modeling_utils import PreTrainedModel

class LoopFormerPretrainedModel(PreTrainedModel):
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ImageEmbeddings):
            if hasattr(module, "position_embeddings"):
                module.position_embeddings.data = nn.init.trunc_normal_(
                    module.position_embeddings.data.to(torch.float32),
                    mean=0.0,
                    std=self.config.initializer_range,
                ).to(module.position_embeddings.dtype)

class LoopFormerForImageClassification(LoopFormerPretrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
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
                        norm_eps=config.layer_norm_eps
                    )
                elif module_type == "mlp":
                    module = Mlp(
                        hidden_size=self.hidden_size,
                        intermediate_size=config.intermediate_size,
                        norm_eps=config.layer_norm_eps
                    )
                elif module_type == "swish_glu":
                    module = SwishGLU(
                        hidden_size=self.hidden_size,
                        intermediate_size=config.intermediate_size,
                        norm_eps=config.layer_norm_eps
                    )
                elif module_type == "identity":
                    module = Identity()
                else:
                    raise ValueError(f"Unknown module type: {module_type}")
                
                self.module_pool.append(module)
        
        # Router for each turn
        if config.router_type == "linear":
            self.router = LinearRouter(
                    hidden_size=self.hidden_size,
                    num_choice=len(self.module_pool)
                )
        elif config.router_type == "mlp":
            self.router = MlpRouter(
                hidden_size=self.hidden_size,
                num_choice=len(self.module_pool),
                router_act=config.router_act,
            )
        elif config.router_type == "linear_with_le":
            self.router = LinearRouterWithLoopEmbedding(
                hidden_size=self.hidden_size,
                num_choice=len(self.module_pool)
            )
        
        self.embeddings = ImageEmbeddings(config)
        self.pooler = Pooler(config)
        self.pool_norm = nn.LayerNorm(config.hidden_size, eps=1e-5, bias=False)
        self.classifier = nn.Linear(config.hidden_size, config.num_classes)
        self.init_weights()

    
    def forward(self, pixel_values: torch.Tensor, num_loops: int = None) -> torch.Tensor:
        """
        Forward pass with router-based module selection
        
        Args:
            hidden_states: Input tensor [B, L, D]
            
        Returns:
            Output tensor [B, L, D]
        """
        hidden_states = self.embeddings(pixel_values)
        num_loops = self.num_loops if num_loops is None else num_loops

        for loop in range(num_loops):
            route_probs = self.router(hidden_states, loop=loop)  # [num_choices]
            idx = torch.argmax(route_probs)  # scalar
            # print(idx.item())
            
            chosen_module = self.module_pool[idx]
            output = chosen_module(hidden_states)
            
            # Residual and update
            hidden_states = hidden_states + output

        hidden_states = self.pool_norm(hidden_states)
        hidden_states = self.pooler(hidden_states)

        return self.classifier(hidden_states) # return the logits
        
