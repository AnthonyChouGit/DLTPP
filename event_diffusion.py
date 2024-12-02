import torch
from torch import nn
from utils import reparametrized_sample
from rounding import embed_rounding

class EventSeqDiffusion(nn.Module):
    def __init__(self, denoise_fn, mark_embed:nn.Embedding, betas, timestep_map=None) -> None:
        super().__init__()
        if isinstance(betas, list):
            betas = torch.tensor(betas, device=mark_embed.weight.device)
        self.timestep_map = timestep_map
        self.timesteps = len(betas)
        self.dim = 1 + mark_embed.embedding_dim
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, device=alpha_cumprod.device), alpha_cumprod[:-1]])
        self.register_buffer('betas', betas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        self.register_buffer('sqrt_alpha_cumprod', torch.sqrt(alpha_cumprod))
        self.register_buffer('sqrt_one_minus_alpha_cumprod', torch.sqrt(1-alpha_cumprod))
        self.register_buffer('sqrt_recip_alpha_cumprod', 1.0/torch.sqrt(alpha_cumprod))
        self.register_buffer('alpha_cumprod_prev', alpha_cumprod_prev)
        self.register_buffer('sqrt_recipm1_alpha_cumprod', torch.sqrt(1.0/alpha_cumprod - 1))
        self.register_buffer('log_one_minus_alpha_cumprod', torch.log(1.0 - alpha_cumprod))
        posterior_variance = \
            betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.maximum(posterior_variance, torch.tensor(1e-20))))
        
        posterior_mean_coef1 = betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        posterior_mean_coef2 = (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod)
        self.register_buffer('posterior_mean_coef1', posterior_mean_coef1)
        self.register_buffer('posterior_mean_coef2', posterior_mean_coef2)

        # Note that our denoising function predicts x0 directly, in order to enable clamping while sampling
        # Not sure whether the clamping trick works in our case
        self.denoise_fn = denoise_fn
        self.embed = mark_embed
        embed_size = mark_embed.embedding_dim
        num_types = mark_embed.num_embeddings
        self.rounding = embed_rounding

        self.mark_net = nn.Sequential(
            nn.Linear(embed_size, 2*embed_size),
            nn.GELU(),
            nn.Linear(2*embed_size, 2*embed_size),
            nn.GELU(),
            nn.Linear(2*embed_size, num_types)
        )

    def _get_logits(self, output_embed):
        logits = torch.log_softmax(self.mark_net(output_embed), dim=-1)
        return logits

    def q_xt_prior(self, x_start, t):
        '''
            x_start: (batch_size, seq_len, embed_size)
            t: (batch_size)
        '''
        batch_size, seq_len, embed_size = x_start.shape
        coef1 = torch.gather(self.sqrt_alpha_cumprod, -1, t) # (batch_size)
        coef2 = torch.gather(self.alpha_cumprod, -1, t) # (batch_size)
        mean = coef1[:, None, None] * x_start
        var = (1-coef2)[:, None, None].expand(batch_size, seq_len, embed_size)
        log_var = torch.gather(self.log_one_minus_alpha_cumprod, -1, t) # (batch_size, )
        log_var = log_var[:, None, None].expand(batch_size, seq_len, embed_size)
        return mean, var, log_var
    
    def q_xtm1_posterior(self, x_start, xt, t):
        '''
            x_start, xt: (batch_size, seq_len, embed_size)
            t: (batch_size)
        '''
        batch_size, seq_len, embed_size = x_start.shape
        x_start_coef = torch.gather(self.posterior_mean_coef1, -1, t) # (batch_size)
        xt_coef = torch.gather(self.posterior_mean_coef2, -1, t) # (batch_size)
        post_mean = x_start_coef[:, None, None] * x_start + xt_coef[:, None, None] * xt
        post_var = torch.gather(self.posterior_variance, -1, t)[:, None, None].expand(batch_size, seq_len, embed_size)
        log_post_var_clipped = torch.gather(self.posterior_log_variance_clipped, -1, t)[:, None, None].expand(batch_size, seq_len, embed_size)
        return post_mean, post_var, log_post_var_clipped
    
    def q_sample(self, x_start, t, noise=None):
        '''
            x_start, xt: (batch_size, seq_len, embed_size)
            t: (batch_size)
        '''
        if noise is None:
            noise = torch.randn_like(x_start)
        xstart_coef = torch.gather(self.sqrt_alpha_cumprod, -1, t) # (batch_size)
        noise_coef = torch.gather(self.sqrt_one_minus_alpha_cumprod, -1, t) # (batch_size)
        sample = xstart_coef[:, None, None] * x_start + noise_coef[:, None, None] * noise
        return sample

    def train_loss(self, mark_seq, log_tau, t, cond): # Padding positions need to be processed before passed in
        '''
            mark_seq/log_tau: (batch_size, seq_len)
            t: (batch_size,)
            cond: (num_layers, batch_size, embed_size)
        '''
        batch_size, seq_len = mark_seq.shape
        embed_size = cond.shape[-1]
        mask = mark_seq.eq(-1)
        mark_seq = mark_seq.masked_fill(mask, 0)
        mark_embed = self.embed(mark_seq)

        x_start_mean = torch.cat([log_tau.unsqueeze(-1), mark_embed], dim=-1)
        mark_start_mean = mark_embed
        mark_start_std = self.sqrt_one_minus_alpha_cumprod[0].expand_as(mark_start_mean)
        mark_start = reparametrized_sample(mark_start_mean, mark_start_std)
        x_start = torch.cat([log_tau.unsqueeze(-1), mark_start], dim=-1)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise) # (batch_size, seq_len, 1+embed_size)
        x_recon = self.denoise_fn(x_t, t, cond) # (batch_size, seq_len, 1+embed_size)
        recon_mse = (x_start - x_recon)**2
        t0_loss = (x_start_mean - x_recon)**2
        t0_mask = (t==0) # (batch_size,)
        recon_mse = torch.where(t0_mask[:, None, None], t0_loss, recon_mse) # (batch_size, seq_len, 1+embed_size)
        
        # theta = 9
        # recon_mse = (1+theta)*recon_mse[..., 0] + (1-theta/embed_size)*recon_mse[..., 1:].sum(-1)
        recon_mse = recon_mse.sum(-1) # (batch_size, seq_len)

        mark_logits = self._get_logits(x_start[..., 1:])

        ce_loss = nn.functional.cross_entropy(mark_logits.transpose(-1, -2), mark_seq.long(), reduction='none') # (batch_size, seq_len)
        T_mean, _, _ = self.q_xt_prior(x_start, torch.tensor([self.timesteps-1], dtype=torch.long, device=x_start.device)) # (batch_size*seq_len, 1+embed_dim)
        tT_loss = (T_mean**2).sum(-1)
        loss = recon_mse + ce_loss + tT_loss # + log_tau_loss
        loss = loss.sum(-1) # (batch_size,)
        return loss

    def p_tm1(self, xt, cond, t, rounding=False):
        x_recon = self.denoise_fn(xt, t, cond)
        if rounding:
            rounded = self.rounding(self.embed.weight, x_recon[..., 1:])
            x_recon = torch.cat([x_recon[..., 0:1], rounded], dim=-1)
        mean, var, log_var = self.q_xtm1_posterior(x_recon, xt, t)
        return mean, var, log_var

    def p_sample(self, xt, cond, t, rounding=False):
        mean, _, log_var = self.p_tm1(xt, cond, t,  rounding)
        noise = torch.randn_like(xt)
        zero_mask = (t==0)
        temp = (0.5 * log_var).exp() * noise
        temp.masked_fill_(zero_mask[:, None, None], 0)
        sample = mean + temp
        return sample
    
    def p_sample_loop(self, cond, pred_len, rounding_start=None):
        batch_size = cond.shape[1]
        x = torch.randn(batch_size, pred_len, 1+self.embed.embedding_dim, device=cond.device)
        for step, i in enumerate(reversed(range(0, self.timesteps))):
            if rounding_start is not None and step>=rounding_start:
                rounding = True
            else:
                rounding = False
            x = self.p_sample(x, cond, torch.full((batch_size,), i, device=cond.device, dtype=torch.long), rounding)
        log_tau_sample = x[..., 0]
        logits = self._get_logits(x[..., 1:])
        mark_dist = torch.distributions.Categorical(logits=logits)
        mark_sample = mark_dist.sample()
        return mark_sample, log_tau_sample
