# Router implementation

import torch
from torch import nn
import math

class LinearRouter(nn.Module):
    """
    A router consisting of a linear layer
    
    Parameters:
        hidden_size: The size of the hidden layer.
        num_choice: The number of choices for the output.  

    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_choice: int = 12,
    ):
        super().__init__()
        self.num_choice = num_choice
        self.hidden_size = hidden_size
        self.r = nn.Linear(hidden_size, num_choice, bias=False) # a simple linear classifier
    

    def forward(self, x: torch.Tensor, loop: int = None) -> torch.Tensor:

        # X: [B, L, D]

        B, L, D = x.shape

        logits = self.r(x)
        probs = torch.softmax(logits, dim=-1)
        
        avg_probs = probs.mean(dim=(0, 1))
        
        return avg_probs

class LinearRouterWithLoopEmbedding(nn.Module):
    """
    A router with sinusoidal loop embedding to avoid repeated identity selection
    
    Parameters:
        hidden_size: The size of the hidden layer.
        num_choice: The number of choices for the output.
        temperature: Temperature for softmax scaling.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_choice: int = 12,
    ):
        super().__init__()
        self.num_choice = num_choice
        self.hidden_size = hidden_size
        
        self.r = nn.Linear(hidden_size, num_choice, bias=False)
    
    def _get_sinusoidal_embedding(self, loop: int, hidden_size: int, device: torch.device) -> torch.Tensor:
        """Generate sinusoidal positional embedding for given loop index"""
        position = torch.tensor(loop, dtype=torch.float, device=device)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float, device=device) *
                           -(math.log(10000.0) / hidden_size))
        
        embedding = torch.zeros(hidden_size, device=device)
        embedding[0::2] = torch.sin(position * div_term)
        embedding[1::2] = torch.cos(position * div_term)
        
        return embedding
    
    def forward(self, x: torch.Tensor, loop: int) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, L, D]
            loop: Current loop index (0-based)
        Returns:
            avg_probs: Average probabilities [num_choice]
        """
        B, L, D = x.shape
        
        # Generate sinusoidal loop embedding
        loop_emb = self._get_sinusoidal_embedding(loop, D, x.device)
        
        # Add loop embedding to input
        x_with_le = x + loop_emb.unsqueeze(0).unsqueeze(0)
        
        # Router forward
        logits = self.r(x_with_le)
        probs = torch.softmax(logits, dim=-1)
        
        avg_probs = probs.mean(dim=(0, 1))
        return avg_probs

class MlpRouter(nn.Module):
    """
    A router consisting of a MLP
    
    Parameters:
        hidden_size: The size of the hidden layer.
        num_choice: The number of choices for the output.  

    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_choice: int = 12,
        intermediate_size: int = None,
        router_act: str = "gelu",
    ):
        super().__init__()
        self.num_choice = num_choice
        self.hidden_size = hidden_size

        if intermediate_size is None:
            self.intermediate_size = hidden_size * 4
        else:
            self.intermediate_size = intermediate_size

        self.r1 = nn.Linear(hidden_size, self.intermediate_size, bias=False)
        self.r2 = nn.Linear(self.intermediate_size, num_choice, bias=False)

        if router_act == "gelu":
            self.act = nn.GELU()
            self.router_act = router_act
        else:
            raise NotImplementedError(f"Router activation {router_act} not implemented")
    

    def forward(self, x: torch.Tensor, loop: int = None) -> torch.Tensor:

        # X: [B, L, D]

        B, L, D = x.shape

        logits = self.r1(x)
        logits = self.act(logits)
        logits = self.r2(logits)
        probs = torch.softmax(logits, dim=-1)
        
        avg_probs = probs.mean(dim=(0, 1))
        
        return avg_probs
            
