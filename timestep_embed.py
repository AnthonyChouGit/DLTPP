import torch
from torch import nn
import torch.nn.functional as F

class TimeStepEmbedding(nn.Module):
    def __init__(self, embed_size, max_steps=1000):
        super().__init__()
        self.register_buffer(
            "embedding", self._build_embedding(embed_size, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(embed_size * 2, embed_size)
        self.projection2 = nn.Linear(embed_size, embed_size)
        self.max_steps = max_steps
        self.embed_size = embed_size

    def forward(self, diffusion_step):
        '''
            diffusion_step: (...)
        '''
        x = self.embedding[diffusion_step] # (..., 2*embed_size)
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x # (..., embed_size)

    def _build_embedding(self, dim, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
        table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table # (T, 2*embed_size)

