import torch
import numpy as np

class LinearSchedule:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=.1) -> None:
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
    
    def get_betas(self):
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        return betas


class CosineScedule:
    def __init__(self, num_timesteps=1000, s=0.008) -> None:
        self.num_timesteps = num_timesteps
        self.s = s

    def get_betas(self):
        timesteps = self.num_timesteps
        steps = timesteps + 1
        s = self.s
        x = np.linspace(0, steps, steps) # (num_timesteps+1)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas =  np.clip(betas, a_min=0, a_max=0.999)
        betas = torch.from_numpy(betas).float()
        return betas