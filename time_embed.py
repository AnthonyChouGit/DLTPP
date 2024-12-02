import torch
from torch import nn

class TrigonoTimeEmbedding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        assert embed_size%2 == 0 
        
        self.Wt = nn.Linear(1, embed_size // 2)
        self.embed_size = embed_size

    def forward(self, interval):
        phi = self.Wt(interval.unsqueeze(-1))
        pe_sin = torch.sin(phi)
        pe_cos = torch.cos(phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)
        return pe
    