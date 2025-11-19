
import torch
import torch.nn.functional as F

def sample_timesteps(scheduler, bsz: int, device: torch.device) -> torch.Tensor:
    num_steps = scheduler.config.num_train_timesteps
    return torch.randint(low=0, high=num_steps, size=(bsz,), device=device, dtype=torch.long)




def construct_noisy_latent_and_etarget(x0, t_int, scheduler, device):
    """
    使用 scheduler 自己的 forward noise 逻辑来保证训练 & 推理一致。
    """
    # 1) 生成 ε
    eps = torch.randn_like(x0, dtype=torch.float32, device=device)

    # 2) 使用 scheduler 内部的 add_noise (不同 scheduler 内部实现不同，但和后向过程配套)
    z_t = scheduler.add_noise(x0.to(torch.float32), eps, t_int)

    # 3) 如果后面还需要 a_bar 来还原 x0，则要按 scheduler 类型计算
    if hasattr(scheduler, "alphas_cumprod"):
        # DDPM/DDIM 类型才有 alphas_cumprod
        a_bar = scheduler.alphas_cumprod.to(device)[t_int].clamp(1e-6, 1 - 1e-6)
    else:
        # 如果是 Euler / Karras 类型，则通过 σ(t) 还原 a_bar：
        # a_bar = 1 / (1 + sigma^2)
        sigma = scheduler.sigmas.to(device)[t_int]
        a_bar = 1.0 / (1.0 + sigma**2)

    return z_t, eps, a_bar


def construct_noisy_latent_and_target(x0, t_int, scheduler, device):
    eps = torch.randn_like(x0, dtype=torch.float32, device=device)
    z_t = scheduler.add_noise(x0.to(torch.float32), eps, t_int)

    # --- 计算 alpha_bar_t ---
    if hasattr(scheduler, "alphas_cumprod"):
        a_bar = scheduler.alphas_cumprod.to(device)[t_int].clamp(1e-6, 1 - 1e-6)
    else:
        sigma = scheduler.sigmas.to(device)[t_int]
        a_bar = 1.0 / (1.0 + sigma**2)

    # --- 计算 target ---
    if scheduler.config.prediction_type == "v_prediction":
        target = (a_bar.sqrt()[:, None, None, None]) * eps - \
                 (1 - a_bar).sqrt()[:, None, None, None] * x0
    else:  # epsilon
        target = eps

    return z_t, target, a_bar




def x0_from_eps(z_t, pred_eps, alpha_bar_t):
    """
    由 ε 预测反推 x0：
        z_t = sqrt(ā) * x0 + sqrt(1-ā) * ε
        => x0 = (z_t - sqrt(1-ā) * ε) / sqrt(ā)
    形状：
        z_t:        [B,4,h,w]
        pred_eps:   [B,4,h,w]
        alpha_bar_t:[B]
    """
    eps = 1e-5
    z_t = z_t.to(torch.float32)
    pred_eps = pred_eps.to(torch.float32)
    alpha_bar_t = alpha_bar_t.to(torch.float32)

    a_sqrt = (alpha_bar_t + eps).sqrt()[:, None, None, None]
    one_minus_a_sqrt = (1.0 - alpha_bar_t + eps).sqrt()[:, None, None, None]

    x0 = (z_t - one_minus_a_sqrt * pred_eps) / a_sqrt  # output fp32
    return x0.clamp(-3, 3)  # 防止极端值

def x0_from_vpred(z_t, pred_v, alpha_bar_t):
    eps = 1e-5
    z_t = z_t.to(torch.float32)
    pred_v = pred_v.to(torch.float32)
    alpha_bar_t = alpha_bar_t.to(torch.float32)
    sqrt_ab = (alpha_bar_t + eps).sqrt()[:, None, None, None]
    sqrt_1mab = (1.0 - alpha_bar_t + eps).sqrt()[:, None, None, None]
    x0 = sqrt_ab * z_t - sqrt_1mab * pred_v
    return x0.clamp(-3, 3)
