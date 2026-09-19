import torch

from attention.scaled_dot_product import ScaledDotProductAttention


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_k=None, d_v=None, dropout=0.1) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.d_v = d_v if d_v is not None else d_model // num_heads

        self.w_q = torch.nn.Linear(d_model, num_heads * self.d_k)
        self.w_k = torch.nn.Linear(d_model, num_heads * self.d_k)
        self.w_v = torch.nn.Linear(d_model, num_heads * self.d_v)
        self.attention = ScaledDotProductAttention(dropout)
        self.fc = torch.nn.Linear(num_heads * self.d_v, d_model)

    def forward(self, query, key, value, mask=None) -> torch.Tensor:
        batch_size = query.size(0)
        query = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        output = self.attention(query, key, value, mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)
        output = self.fc(output)
        return output
