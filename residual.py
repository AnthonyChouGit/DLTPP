import torch
from torch import nn
import math
import torch.nn.functional as F
from timestep_embed import TimeStepEmbedding

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, cond_size, dilation, embed_size, use_step=True):
        super().__init__()
        # self.step_embed = step_embed
        if use_step:
            self.step_proj = nn.Linear(embed_size, residual_channels)
        self.cond_proj = nn.Conv1d(cond_size, 2*residual_channels, 1)
        self.dilated_conv = nn.Conv1d(residual_channels, 2*residual_channels, 3, padding=dilation, dilation=dilation)
        self.out_proj = nn.Conv1d(residual_channels, 2*residual_channels, 1)
        self.use_step = use_step

        nn.init.kaiming_normal_(self.cond_proj.weight)
        nn.init.kaiming_normal_(self.out_proj.weight)
        self.norm = nn.LayerNorm(residual_channels)

    def forward(self, x, cond, step_embed):
        '''
            x: (batch_size, seq_len, residual_channels)
            cond: (batch_size, seq_len, cond_size)
            step: (batch_size, embed_size)
        '''

        # step_embed = self.step_embed(step) # (batch_size, embed_size)
        if self.use_step:
            step_embed = self.step_proj(step_embed) # (batch_size, residual_channels)
        x = x.transpose(-1, -2) # (batch_size, residual_channels, seq_len)
        y = x + step_embed[:, :, None]
        cond = cond.transpose(-1, -2) # (batch_size, cond_size, seq_len)
        cond = self.cond_proj(cond) # (batch_size, 2*residual_channels, seq_len)
        # y = self.norm(y.transpose(-1, -2)).transpose(-1, -2)

        y = self.dilated_conv(y) + cond # (batch_size, 2*residual_channels, seq_len)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.out_proj(y)
        y = F.silu(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        out = (x+residual)/math.sqrt(2.)
        out = out.transpose(-1, -2)
        skip = skip.transpose(-1, -2)
        return out, skip

class ResidualNet(nn.Module):
    def __init__(self, input_size, output_size, residual_channels, cond_size, step_embed: TimeStepEmbedding, embed_size, layer_num=2, dilation_cycle_length=2) -> None:
        super().__init__()
        if step_embed is not None:
            assert embed_size == step_embed.embed_size
            use_step = True
        else:
            use_step = False
        
        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels, cond_size, 2 ** (i % dilation_cycle_length), embed_size, use_step)
            for i in range(layer_num)
        ])
        self.skip_projection = nn.Conv1d(residual_channels, residual_channels, 3, padding=1)
        self.input_proj = nn.Linear(input_size, residual_channels)
        self.output_proj = nn.Conv1d(residual_channels, output_size, 3, padding=1)
        nn.init.kaiming_normal_(self.skip_projection.weight)
        self.step_embed = step_embed
        self.embed_size = embed_size
        # self.norm1 = nn.LayerNorm(residual_channels)
        # self.norm2 = nn.LayerNorm(residual_channels)

    def forward(self, x, step, cond):
        '''
            x: (batch_size, seq_len, input_size)
            cond: (batch_size, seq_len, cond_size)
            step: (batch_size, )
        '''
        x = self.input_proj(x) # (batch_size, seq_len, residual_channels)
        x = F.silu(x)
        skip = list()
        for layer in self.residual_layers:
            if self.step_embed is not None:
                step_embed = self.step_embed(step)
            else:
                step_embed = torch.zeros(self.embed_size, device=cond.device)
            x, skip_value = layer(x, cond, step_embed) # x: (batch_size, seq_len, residual_channels)
            skip.append(skip_value)
        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers)) # (batch_size, seq_len, residual_channels)
        # x = self.norm1(x)
        x = self.skip_projection(x.transpose(-1, -2)).transpose(-1, -2)
        x = F.silu(x)
        # x = self.norm2(x)
        x = self.output_proj(x.transpose(-1, -2)).transpose(-1, -2)
        return x

