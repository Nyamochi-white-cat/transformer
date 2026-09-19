import torch

from attention.masks import causal_mask, combine_masks


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, dropout=0.1) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, key, value, custom_mask=None, causal: bool = False) -> torch.Tensor:
        d_k = query.size(-1)
        if d_k != key.size(-1):
            raise ValueError("Query and key must have the same dimension")
        weight = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(d_k, dtype=torch.float32)
        )
        masks = []
        if causal:
            masks.append(causal_mask(weight.size(-1)).to(weight.device))
        if custom_mask is not None:
            masks.append(custom_mask)
        if masks:
            combined_mask = combine_masks(*masks)
            weight = weight.masked_fill(combined_mask == 0, float("-inf"))
        weight = self.softmax(weight)
        weight = self.dropout(weight)
        assert value.size(-2) == weight.size(-1), (
            "The num of sequence in value must be equal to the num of sequence in query and key"
        )
        output = torch.matmul(weight, value)
        return output
