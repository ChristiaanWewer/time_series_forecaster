import torch
from .soft_dtw import soft_dtw_value, soft_dtw_alignment


def dilate_loss(outputs, targets, alpha, gamma, device):
    # outputs, targets: shape (batch_size, N_output, 1)
    batch_size, N_output = outputs.shape[0:2]

    t = targets[:, :, 0]  # (B, N_output)
    o = outputs[:, :, 0]  # (B, N_output)
    D = (t.unsqueeze(2) - o.unsqueeze(1)).pow(2)  # (B, N_output, N_output)

    sdtw_val, R = soft_dtw_value(D, gamma)
    loss_shape = sdtw_val.mean()

    path = soft_dtw_alignment(D.detach(), R.detach(), gamma)  # (B, N_output, N_output)
    idx = torch.arange(1, N_output + 1, device=device, dtype=outputs.dtype)
    Omega = (idx.unsqueeze(1) - idx.unsqueeze(0)).pow(2)  # (N_output, N_output)
    loss_temporal = (path.mean(dim=0) * Omega).sum() / (N_output * N_output)

    loss = alpha * loss_shape + (1 - alpha) * loss_temporal
    return loss, loss_shape, loss_temporal
