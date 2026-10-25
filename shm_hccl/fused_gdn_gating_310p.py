import torch
import torch.nn.functional as F


def fused_gdn_gating_pytorch(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    neg_exp_A = getattr(A_log, "_neg_exp_fp32", None)
    if neg_exp_A is None or neg_exp_A.device != A_log.device:
        neg_exp_A = (-torch.exp(A_log.to(torch.float32))).unsqueeze(0)
        try:
            A_log._neg_exp_fp32 = neg_exp_A
        except Exception:
            pass

    dt_bias_f = getattr(dt_bias, "_fp32_2d", None)
    if dt_bias_f is None or dt_bias_f.device != dt_bias.device:
        dt_bias_f = dt_bias.to(torch.float32).unsqueeze(0)
        try:
            dt_bias._fp32_2d = dt_bias_f
        except Exception:
            pass

    x = a.to(torch.float32) + dt_bias_f
    if beta == 1.0:
        softplus_x = F.softplus(x, beta=1.0, threshold=threshold)
    else:
        beta_x = beta * x
        softplus_x = torch.where(
            beta_x <= threshold,
            (1.0 / beta) * torch.log1p(torch.exp(beta_x)),
            x,
        )
    g = (neg_exp_A * softplus_x).unsqueeze(0)
    beta_output = torch.sigmoid(b.to(torch.float32)).to(b.dtype).unsqueeze(0)
    return g, beta_output
