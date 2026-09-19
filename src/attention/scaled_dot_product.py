import torch


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout=0.1) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None) -> torch.Tensor:
        d_k = query.size(-1)
        if d_k != key.size(-1):
            raise ValueError("Query and key must have the same dimension")
        weight = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )
        if mask is not None:
            if mask.size(-1) != weight.size(-1) or mask.size(-2) != weight.size(-2):
                raise ValueError("Mask size must match the size of the weight matrix")
            weight = weight.masked_fill(mask == 0, float("-inf"))
        weight = self.softmax(weight)
        weight = self.dropout(weight)
        assert value.size(-2) == weight.size(-1), (
            "The num of sequence in value must be equal to the num of sequence in query and key"
        )
        output = torch.matmul(weight, value)
        return output
