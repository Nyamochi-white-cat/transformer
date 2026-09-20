import torch

from attention.multi_head import MultiHeadAttention
from layers.feed_forward import FeedForward
from layers.normalization import LayerNorm


class EncoderBlock(torch.nn.Module):
    def __init__(self, d_model=512, head_num=8, d_ff=2048):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, head_num)
        self.layer_norm_1 = LayerNorm(d_model)
        self.layer_norm_2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor):
        x = x + self.mha(x, x, x)
        x = self.layer_norm_1(x)
        x = x + self.feed_forward(x)
        return self.layer_norm_2(x)


class EncoderStack(torch.nn.Module):
    def __init__(self, num_layers=6, d_model=512, head_num=8, d_ff=2048):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [EncoderBlock(d_model, head_num, d_ff) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
