import torch
from torch import nn
from event_diffusion import EventSeqDiffusion
from denoising import ResidualDenoise_Fn, RNNDenoise_Fn, MLPDenoise_Fn
from time_embed import TrigonoTimeEmbedding
from timestep_sampler import LossAwareSampler
from beta_schedule import LinearSchedule, CosineScedule
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from utils import normalize, unnormalize

class DLTPP(nn.Module):
    def __init__(self, 
        config,
        event_type_num: int,
        embed_size: int = 32,
        max_dt: float=50.0,
                 ) -> None:
        super().__init__()
        train_steps = config['train_steps']
        self.rounding_start = config['rounding_start']
        self.block_num = config['block_num']
        self.weight_decay = config['weight_decay']
        self.beta_schedule = CosineScedule(train_steps)
        self.encoder = nn.GRU(input_size=1+embed_size, hidden_size=embed_size, batch_first=True)
        # self.time_embed = TrigonoTimeEmbedding(embed_size)
        if 'denoise' not in config or config['denoise']=='residual':
            self.denoise_fn = ResidualDenoise_Fn(embed_size, 1+embed_size, embed_size, train_steps, self.block_num)
        elif config['denoise']=='rnn':
            self.denoise_fn = RNNDenoise_Fn(embed_size, 1+embed_size, train_steps)
        elif config['denoise']=='mlp':
            self.denoise_fn = MLPDenoise_Fn(embed_size, 1+embed_size, train_steps)
        self.embed = nn.Embedding(event_type_num, embed_size)
        self.dec_embed = nn.Embedding(event_type_num, embed_size)
        self.diffusion = EventSeqDiffusion(self.denoise_fn, self.dec_embed, self.beta_schedule.get_betas())
        self.optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=self.weight_decay)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma=0.999)
        self.steps = train_steps


        # self.timestep_sampler = LossAwareSampler(train_steps)

    def encode(self, seq_dts, seq_marks):
        mask = seq_marks.eq(-1)
        seq_marks = seq_marks.masked_fill(mask, 0)
        type_embed = self.embed(seq_marks)
        # time_embed = self.time_embed(seq_dts)
        log_tau = torch.log(seq_dts.clamp(1e-8))
        log_tau = normalize(log_tau, self.log_mean, self.log_std)
        event_embed = torch.cat([log_tau.unsqueeze(-1), type_embed], dim=-1)
        hist_embed, hist_states = self.encoder(event_embed)
        return hist_embed, hist_states
    
    def forward(self, seq_dts, seq_marks, hist_len, pred_len, stride):
        batch_size, seq_len = seq_marks.shape
        window_size = hist_len + pred_len
        window_num = (seq_len-window_size) // stride + 1
        window_start = torch.arange(start=0, end=window_num*stride, step=stride, device=seq_dts.device)
        window_end = window_start + window_size
        mask = seq_marks.eq(-1)
        assert window_num == len(window_start)
        total_loss = torch.tensor(0, device=seq_dts.device)
        event_num = 0
        for i in range(window_num):
            hist_dts = seq_dts[:, window_start[i]:window_start[i]+hist_len]
            pred_dts = seq_dts[:, window_start[i]+hist_len:window_end[i]]
            hist_marks = seq_marks[:, window_start[i]:window_start[i]+hist_len]
            pred_marks = seq_marks[:, window_start[i]+hist_len:window_end[i]]
            window_mask = mask[:, window_start[i]:window_end[i]]
            temp = window_mask.sum(1)
            window_mask = (temp>0)
            _, hist_states = self.encode(hist_dts, hist_marks)
            pred_log_tau = torch.log(pred_dts.clamp(1e-8))
            pred_log_tau = normalize(pred_log_tau, self.log_mean, self.log_std)
            t, weight = self.timestep_sampler.sample(batch_size)
            t, weight = t.to(seq_dts.device), weight.to(seq_dts.device)
            loss = self.diffusion.train_loss(pred_marks, pred_log_tau, t, hist_states) * weight
            loss.masked_fill_(window_mask, 0)
            loss_for_update = loss.masked_select(~window_mask)
            ts_for_update = t.masked_select(~window_mask)
            self.timestep_sampler.update_loss(ts_for_update, loss_for_update)
            total_loss = total_loss + loss.sum()
            event_num += (~window_mask).sum()*pred_len
        return total_loss, event_num
    
    def train_self(self, seq_dts, seq_marks, hist_len, pred_len, stride):
        batch_size, seq_len = seq_marks.shape
        window_size = hist_len + pred_len
        window_num = (seq_len-window_size) // stride + 1
        window_start = torch.arange(start=0, end=window_num*stride, step=stride, device=seq_dts.device)
        window_end = window_start + window_size
        mask = seq_marks.eq(-1)
        assert window_num == len(window_start)
        for i in range(window_num):
            self.optimizer.zero_grad()
            hist_dts = seq_dts[:, window_start[i]:window_start[i]+hist_len]
            pred_dts = seq_dts[:, window_start[i]+hist_len:window_end[i]]
            hist_marks = seq_marks[:, window_start[i]:window_start[i]+hist_len]
            pred_marks = seq_marks[:, window_start[i]+hist_len:window_end[i]]
            window_mask = mask[:, window_start[i]:window_end[i]]
            temp = window_mask.sum(1)
            window_mask = (temp>0)
            _, hist_states = self.encode(hist_dts, hist_marks)
            pred_log_tau = torch.log(pred_dts.clamp(1e-8))
            pred_log_tau = normalize(pred_log_tau, self.log_mean, self.log_std)
            # t, weight = self.timestep_sampler.sample(batch_size)
            # t, weight = t.to(seq_dts.device), weight.to(seq_dts.device)
            t = torch.randint(0, self.steps, (batch_size,), device=hist_dts.device)
            loss = self.diffusion.train_loss(pred_marks, pred_log_tau, t, hist_states) # * weight
            loss.masked_fill_(window_mask, 0)
            # with torch.no_grad():
            #     loss_for_update = loss.masked_select(~window_mask)
            #     ts_for_update = t.masked_select(~window_mask)
            #     self.timestep_sampler.update_loss(ts_for_update, loss_for_update)
            loss = loss.sum()
            loss.backward()
            self.optimizer.step()
        self.lr_scheduler.step()


    @torch.no_grad()
    def predict(self, hist_dts, hist_marks, pred_len, sample_num=200): 
        _, hist_states = self.encode(hist_dts, hist_marks) # (num_layers, batch_size, embed_size)
        num_layers, batch_size, embed_size = hist_states.shape
        hist_states = hist_states[:, :, None, :].expand(num_layers, batch_size, sample_num, embed_size).contiguous()
        hist_states = hist_states.view(num_layers, batch_size*sample_num, embed_size)
        mark_sample, log_tau_sample = self.diffusion.p_sample_loop(hist_states, pred_len, rounding_start=self.rounding_start) # (batch_size*sample_num, pred_len)
        log_tau_sample = unnormalize(log_tau_sample, self.log_mean, self.log_std)
        dt_sample = log_tau_sample.exp()
        dt_sample.clamp_(max=self.max_dt)
        t_sample = torch.cumsum(dt_sample, dim=1) # (batch_size*sample_num, pred_len)
        t_sample = t_sample.view(batch_size, sample_num, pred_len)
        mark_sample = mark_sample.view(batch_size, sample_num, pred_len)
        return mark_sample, t_sample
