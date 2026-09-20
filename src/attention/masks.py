import torch


def causal_mask(seq_len: int) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)


def padding_mask(token_ids: torch.Tensor, padding_id: int) -> torch.Tensor:
    return token_ids.ne(padding_id).unsqueeze(1).unsqueeze(2)


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    if not masks:
        return torch.ones(1, 1, 1, 1, dtype=torch.bool)
    combined_mask = torch.ones_like(masks[0], dtype=torch.bool)
    for mask in masks[1:]:
        combined_mask = combined_mask & mask
    return combined_mask
