import torch

def reparametrized_sample(mean, std):
    assert mean.shape == std.shape
    eps = torch.randn_like(mean)
    sample = mean + eps * std
    return sample

def normalize(input_value, mean, std):
    out = (input_value - mean) / std
    return out

def unnormalize(input_value, mean, std):
    out = input_value * std + mean
    return out

