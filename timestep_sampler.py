import torch
from abc import ABC, abstractmethod

class TimeStepSampler(ABC):
    @abstractmethod
    def weights(self):
        pass

    def sample(self, batch_size):
        w = self.weights() # (num_timesteps,)
        p = w / w.sum()
        dist = torch.distributions.Categorical(probs=p)
        inds = dist.sample([batch_size]) # (batch_size,)
        scale_weights = 1 / (len(p) * p[inds])
        return inds, scale_weights

class LossAwareSampler(TimeStepSampler):
    def __init__(self, num_timesteps, history_per_step=10, uniform_prob=.001) -> None:
        self.num_timesteps = num_timesteps
        self.history_per_step = history_per_step
        self.uniform_prob = uniform_prob

        self.loss_history = torch.zeros(num_timesteps, history_per_step)
        self.loss_counts = torch.zeros(num_timesteps, dtype=torch.int)

    def weights(self):
        if not self._warmed_up():
            return torch.ones(self.num_timesteps)
        weights = torch.sqrt(torch.mean(self.loss_history**2, dim=-1))
        weights /= weights.sum()
        weights *= (1 - self.uniform_prob)
        weights += (self.uniform_prob/len(weights))
        return weights
    
    @torch.no_grad()
    def update_loss(self, ts, losses): 
        for t, loss in zip(ts, losses):
            if self.loss_counts[t] == self.history_per_step:
                self.loss_history[t, :-1] = self.loss_history[t, 1:].clone()
                self.loss_history[t, -1] = loss
            else:
                self.loss_history[t, self.loss_counts[t]] = loss
                self.loss_counts[t] += 1

    def _warmed_up(self):
        flag = torch.all(self.loss_counts==self.history_per_step)
        return flag