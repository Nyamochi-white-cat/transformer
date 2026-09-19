import torch

from attention.multi_head import MultiHeadAttention
from layers.feed_forward import FeedForward
from layers.normalization import LayerNorm
from layers.positional import positional_encoding


class DecoderBlock(torch.nn.Module):
    def __init__(self, d_model=512, head_num=8, d_ff=2048):
        super().__init__()
        self.mha_1 = MultiHeadAttention(d_model, head_num)
        self.layer_norm_1 = LayerNorm(d_model)
        self.mha_2 = MultiHeadAttention(d_model, head_num)
        self.layer_norm_2 = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm_3 = LayerNorm(d_model)

    def forward(self, x, encoder_output, custom_mask=None):
        x = x + self.mha_1(x, x, x, causal=True)
        x = self.layer_norm_1(x)
        x = x + self.mha_2(x, encoder_output, encoder_output, custom_mask=custom_mask)
        x = self.layer_norm_2(x)
        x = x + self.feed_forward(x)
        return self.layer_norm_3(x)


class Decoder(torch.nn.Module):
    def __init__(self, num_layers=6, d_model=512, head_num=8, d_ff=2048):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [DecoderBlock(d_model, head_num, d_ff) for _ in range(num_layers)]
        )
        self.w = torch.nn.Linear(d_model, d_model)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, encoder_output, custom_mask=None):
        x += positional_encoding(x.size(1), x.size(2)).to(x.device)
        for layer in self.layers:
            x = layer(x, encoder_output, custom_mask)
        x = self.w(x)
        x = self.softmax(x)
        return x
