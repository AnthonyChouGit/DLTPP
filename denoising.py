import torch
from torch import nn
from residual import ResidualNet
from timestep_embed import TimeStepEmbedding
from time_embed import TrigonoTimeEmbedding

class ResidualDenoise_Fn(nn.Module):
    def __init__(self, embed_size, output_size, residual_channels, max_steps=1000, residual_layers=2) -> None:
        super().__init__()
        self.time_embed = TrigonoTimeEmbedding(embed_size)
        input_size = 1 + 2*embed_size
        self.step_embed = TimeStepEmbedding(embed_size, max_steps)
        self.sequential_net = nn.GRU(input_size=input_size, hidden_size=embed_size, batch_first=True)
        self.residual_net = ResidualNet(input_size, embed_size, residual_channels, embed_size, self.step_embed, embed_size, residual_layers)
        # self.output_proj = nn.Linear(2*embed_size, output_size)
        self.out_proj1 = nn.Sequential(
            nn.Linear(embed_size, 2*embed_size),
            nn.GELU(),
            nn.Linear(2*embed_size, output_size)
        )

        self.dim = 1+embed_size

    def forward(self, x, t, cond):
        trigono_embed = self.time_embed(x[..., 0].exp())
        feature_input = torch.cat([x, trigono_embed], dim=-1)
        res_cond, _ = self.sequential_net(feature_input, cond)
        res_out = self.residual_net(feature_input, t, res_cond)

        # temp = torch.cat([res_out, res_cond], dim=-1)
        # x_recon = self.output_proj(temp)
        x_recon = self.out_proj1(res_out + res_cond)

        return x_recon

class RNNDenoise_Fn(nn.Module):
    def __init__(self, embed_size, output_size, max_steps=1000) -> None:
        super().__init__()
        self.time_embed = TrigonoTimeEmbedding(embed_size)
        input_size = 1 + 2*embed_size
        self.step_embed = TimeStepEmbedding(embed_size, max_steps)
        self.sequential_net = nn.GRU(input_size=input_size, hidden_size=embed_size, batch_first=True)
        # self.output_proj = nn.Linear(2*embed_size, output_size)
        self.out_proj1 = nn.Sequential(
            nn.Linear(embed_size, 2*embed_size),
            nn.GELU(),
            nn.Linear(2*embed_size, output_size)
        )

        self.dim = 1+embed_size
        

    def forward(self, x, t, cond):
        trigono_embed = self.time_embed(x[..., 0].exp())
        feature_input = torch.cat([x, trigono_embed], dim=-1)
        hidden, _ = self.sequential_net(feature_input, cond)
        step_embed = self.step_embed(t)
        
        hidden = hidden + step_embed[:, None, :]
        x_recon = self.out_proj1(hidden)
        
        # temp = torch.cat([res_out, res_cond], dim=-1)
        # x_recon = self.output_proj(temp)
        # x_recon = self.out_proj1(res_out + res_cond)

        return x_recon

class MLPDenoise_Fn(nn.Module):
    def __init__(self, embed_size, output_size, max_steps=1000) -> None:
        super().__init__()
        self.time_embed = TrigonoTimeEmbedding(embed_size)
        input_size = 1 + 2*embed_size
        self.step_embed = TimeStepEmbedding(embed_size, max_steps)
        self.input_proj = nn.Linear(input_size, embed_size)
        self.mlp_net = nn.Sequential(
            nn.Linear(2*embed_size, 2*embed_size),
            nn.GELU(),
            nn.Linear(2*embed_size, output_size)
        )
        self.dim = 1+embed_size

    def forward(self, x, t, cond):
        batch_size, seq_len, _ = x.shape
        trigono_embed = self.time_embed(x[..., 0].exp())
        feature_input = torch.cat([x, trigono_embed], dim=-1)
        feature = self.input_proj(feature_input)
        step_embed = self.step_embed(t)
        feature = feature + step_embed[:, None, :]
        feature = torch.cat([feature, cond.squeeze(0)[:, None, :].expand(batch_size, seq_len, self.dim-1)], dim=-1)
        x_recon = self.mlp_net(feature)
        return x_recon