import torch


class GaussianSampler:
    def __init__(self, parameters, deterministic=True):
        self.parameters = parameters
        self.mu, self.log_var = torch.chunk(parameters, 2, dim=1)

    def sample(self):
        return self.mu + torch.exp(0.5 * self.log_var) * torch.randn_like(self.mu)

    def kl(self):
        kl_loss = -0.5 * torch.sum(1 + self.log_var - self.mu.pow(2) - self.log_var.exp()) / self.mu.shape[0]

        return kl_loss
