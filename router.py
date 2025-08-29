# Router implementation

import torch
from torch import nn

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
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # X: [B, L, D]

        B, L, D = x.shape

        logits = self.r(x)
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

        self.r1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.r2 = nn.Linear(intermediate_size, num_choice, bias=False)

        if router_act == "gelu":
            self.act = nn.GELU()
            self.router_act = router_act
        else:
            raise NotImplementedError(f"Router activation {router_act} not implemented")
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # X: [B, L, D]

        B, L, D = x.shape

        logits = self.r1(x)
        logits = self.act(logits)
        logits = self.r2(logits)
        probs = torch.softmax(logits, dim=-1)
        
        avg_probs = probs.mean(dim=(0, 1))
        
        return avg_probs
            
