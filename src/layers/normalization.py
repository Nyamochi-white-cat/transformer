import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(d_model, requires_grad=True)
        self.beta = torch.zeros(d_model, requires_grad=True)

    def forward(self, x: torch.Tensor, dim=-1):
        mu = x.mean(dim)
        sigma = torch.sqrt(x.var(dim) + self.eps)
        return torch.matmul(self.gamma, (x - mu) / sigma) + self.beta
