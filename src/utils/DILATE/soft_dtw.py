import torch


def soft_dtw_value(D, gamma):
    """Forward soft-DTW DP over (B, N, M) distance matrix. Fully differentiable via autograd."""
    B, N, M = D.shape
    R = torch.full((B, N + 2, M + 2), float('inf'), device=D.device, dtype=D.dtype)
    R[:, 0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            stk = torch.stack([R[:, i - 1, j - 1], R[:, i - 1, j], R[:, i, j - 1]], dim=1)
            R[:, i, j] = D[:, i - 1, j - 1] - gamma * torch.logsumexp(-stk / gamma, dim=1)
    return R[:, N, M], R  # (B,), (B, N+2, M+2)


@torch.no_grad()
def soft_dtw_alignment(D, R, gamma):
    """Backward DP: soft alignment path E. D and R must be detached."""
    B, N, M = D.shape
    D_p = torch.zeros(B, N + 2, M + 2, device=D.device, dtype=D.dtype)
    D_p[:, 1:N + 1, 1:M + 1] = D
    R = R.clone()
    R[:, :, M + 1] = -1e8
    R[:, N + 1, :] = -1e8
    R[:, N + 1, M + 1] = R[:, N, M]  # terminal: exp((R[N,M]-R[N,M]-0)/gamma) = 1
    E = torch.zeros(B, N + 2, M + 2, device=D.device, dtype=D.dtype)
    E[:, N + 1, M + 1] = 1.0
    for i in range(N, 0, -1):
        for j in range(M, 0, -1):
            a = torch.exp((R[:, i + 1, j    ] - R[:, i, j] - D_p[:, i + 1, j    ]) / gamma)
            b = torch.exp((R[:, i,     j + 1] - R[:, i, j] - D_p[:, i,     j + 1]) / gamma)
            c = torch.exp((R[:, i + 1, j + 1] - R[:, i, j] - D_p[:, i + 1, j + 1]) / gamma)
            E[:, i, j] = E[:, i + 1, j] * a + E[:, i, j + 1] * b + E[:, i + 1, j + 1] * c
    return E[:, 1:N + 1, 1:M + 1]
