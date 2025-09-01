import torch

from torch import nn

class Mixer(nn.Module):
    """
    A general mixer class for a combination of token and channel mixers
    """
    def __init__(
            self,
            token_mixer: str, 
            channel_mixer: str,
        ):
        
        super().__init__()
        self.token_mixer = token_mixer
        self.channel_mixer = channel_mixer
    
    def forward(self, x):
        # with residual
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x