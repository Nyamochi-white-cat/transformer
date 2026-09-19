import torch


def positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)
    angle = pos / torch.pow(10000, (2 * (i // 2)) / d_model)
    angle[:, 0::2] = torch.sin(angle[:, 0::2])
    angle[:, 1::2] = torch.cos(angle[:, 1::2])
    return angle
