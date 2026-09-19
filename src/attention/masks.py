import torch


def causal_mask(seq_len: int) -> torch.Tensor:
    return torch.tril(torch.ones(seq_len, seq_len))


def padding_mask(seq_len: int, padding_idx: int) -> torch.Tensor:
    mask = torch.ones(seq_len, seq_len)
    mask[padding_idx:, :] = 0
    return mask


def combine_masks(*masks: torch.Tensor) -> torch.Tensor:
    combined_mask = torch.ones_like(masks[0])
    assert all(mask.size() == combined_mask.size() for mask in masks), (
        "All masks must have the same size"
    )
    for mask in masks:
        combined_mask = combined_mask * mask
    return combined_mask
