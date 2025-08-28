# Router implementation

import torch
from torch import nn

class Router(nn.Module):
    """
    A router
    
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
