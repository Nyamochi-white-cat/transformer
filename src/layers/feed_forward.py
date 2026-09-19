import torch

from layers.activate import relu


class FeedForward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = torch.nn.Linear(d_model, d_ff)
        self.w2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(relu(self.w1(x)))
