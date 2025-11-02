# cond_modules.py
# Modules for SD×4 + FRI + DS-TCM (+ optional QRM)
# PyTorch 2.x, diffusers >= 0.26

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

from utils.loss import BandConsistencyLoss, LapPyramid

__all__ = [
    "timestep_embedding",
    "compute_snr",
    "BandConsistencyLoss",
    "SDX4_FRI_DSTCM_Wrapper",
]

class FlowBackend(nn.Module):
    """
    forward(lr_j:[B,3,H,W] in [0,1], lr_i:[B,3,H,W] in [0,1]) -> flow_lr_{j→i}:[B,2,H,W]
    """
    def __init__(self):
        super().__init__()

    def forward(self, lr_j: torch.Tensor, lr_i: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError



# ========== Basics ==========
def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10_000):
    half = dim // 2
    maxp = torch.tensor(max_period, device=timesteps.device, dtype=torch.float32)
    freqs = torch.exp(
        -torch.arange(0, half, dtype=torch.float32, device=timesteps.device)
        * (torch.log(maxp) / half)
    )

    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def compute_snr(alpha_bar: torch.Tensor, eps: float = 1e-8):
    return alpha_bar / (1.0 - alpha_bar + eps)


class ZeroConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch):
        super().__init__(in_ch, out_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.weight)
        nn.init.zeros_(self.bias)



def make_tiny_enc(in_ch: int, out_ch: int):
    # 自动选择能整除的分组数，优先 8，其次 4/2/1
    def _gn(c: int):
        for g in (8, 4, 2, 1):
            if c % g == 0:
                return nn.GroupNorm(g, c)
        return nn.GroupNorm(1, c)

    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        _gn(out_ch),
        nn.SiLU(),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        _gn(out_ch),
        nn.SiLU(),
    )


# ========== DS-TCM ==========
class DSTCMGate(nn.Module):
    def __init__(self, emb_dim: int, feat_dims: Dict[str, int], use_snr=True):
        super().__init__()
        self.use_snr = use_snr
        in_dim = emb_dim * (2 if use_snr else 1)
        self.mlps = nn.ModuleDict({k: nn.Linear(in_dim, 2 * c) for k, c in feat_dims.items()})
        self.high_gate = nn.Linear(in_dim, feat_dims["high"])

    def forward(self, t_emb, snr_emb, feats: Dict[str, torch.Tensor]):
        # FIX: 强制统一 dtype 到 feats 的 dtype（AMP 下 Linear 为 fp32）
        base_dtype = next(iter(feats.values())).dtype
        x = torch.cat([t_emb, snr_emb], dim=-1) if self.use_snr else t_emb
        x = x.to(dtype=base_dtype)

        out = {}
        for k, Fk in feats.items():
            g, b = self.mlps[k](x).chunk(2, dim=-1)
            g, b = g[..., None, None], b[..., None, None]
            out[k] = g * Fk + b
        gate = torch.sigmoid(self.high_gate(x))[..., None, None]
        out["high"] = out["high"] * gate
        return out



class QRM(nn.Module):
    """质量路由：为每个 warped 参考生成 per-pixel gate."""
    def __init__(self, in_ch=6, device='cpu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )
        self.device = device
        self.to(device)
        
        
        
    def forward(self, hr_j2i: torch.Tensor, lr_i_up: torch.Tensor):
        # NOTE: 这里假设 hr_j2i 与 lr_i_up 分辨率一致
        x = torch.cat([hr_j2i, lr_i_up], dim=1)
        # FIX: AMP 下保持与输入相同的 dtype
        x = x.to(dtype=hr_j2i.dtype)
        return self.net(x)



# ========== FRI + DS-TCM conditioner ==========
@dataclass
class ScaleSpec:
    ch: int
    factor: int  # e.g., 8,4,2


class FRI_DSTCM_Conditioner(nn.Module):
    def __init__(self, n_rgbk: int, scale_specs: Dict[str, ScaleSpec]):
        super().__init__()
        self.n_rgbk = n_rgbk
        self.lap = LapPyramid(n_rgbk)

        total_out = sum(s.ch for s in scale_specs.values())
        hid = max(32, total_out // 4)

        self.enc_low = make_tiny_enc(n_rgbk, hid)
        self.enc_mid = make_tiny_enc(n_rgbk, hid)
        self.enc_high = make_tiny_enc(n_rgbk, hid)

        self.scales = scale_specs
        self.proj_low = nn.ModuleDict()
        self.proj_mid = nn.ModuleDict()
        self.proj_high = nn.ModuleDict()
        for name, spec in self.scales.items():
            self.proj_low[name] = ZeroConv2d(hid, spec.ch)
            self.proj_mid[name] = ZeroConv2d(hid, spec.ch)
            self.proj_high[name] = ZeroConv2d(hid, spec.ch)

        feat_dims = {"low": hid, "mid": hid, "high": hid}
        self.ds_gate = DSTCMGate(emb_dim=256, feat_dims=feat_dims, use_snr=True)

    def _resize(self, x, factor):
        return x if factor == 1 else F.interpolate(x, scale_factor=1.0 / factor, mode="bilinear", align_corners=False)

    def forward(self, cond_i, t_emb, snr_emb):
        t_emb  = t_emb.to(dtype=cond_i.dtype)
        snr_emb = snr_emb.to(dtype=cond_i.dtype)

        low, mid, high = self.lap(cond_i)
        F_low = self.enc_low(low)
        F_mid = self.enc_mid(mid)
        F_high = self.enc_high(high)
        feats = self.ds_gate(t_emb, snr_emb, {"low": F_low, "mid": F_mid, "high": F_high})

        residuals = {}
        for name, spec in self.scales.items():
            fl = self._resize(feats["low"], spec.factor)
            fm = self._resize(feats["mid"], spec.factor)
            fh = self._resize(feats["high"], spec.factor)
            residuals[name] = self.proj_low[name](fl) + self.proj_mid[name](fm) + self.proj_high[name](fh)
        return residuals

class UNetMultiScaleInjector:
    def __init__(self, unet, scale_keys: List[str]):
        self.unet = unet
        self.scale_keys = scale_keys  # e.g. ["s2","s4","s8"]
        self._probes, self._shapes, self._hooks = [], {}, []
        self._proj_convs = nn.ModuleDict()  # ★ 新增：保存每个注入点的1x1卷积映射器

    def _probe_hook(self, name):
        def fn(module, inputs):
            x = inputs[0]
            self._shapes[name] = x.shape[-2:]  # 只记录H,W
        return fn

    def probe(self):
        for i, up in enumerate(self.unet.up_blocks):
            for j, res in enumerate(up.resnets):
                h = res.register_forward_pre_hook(self._probe_hook(f"up{i}_res{j}"))
                self._probes.append(h)

    def resolve(self):
        # 分辨率从高到低排序，对应 scale_keys = ["s2","s4","s8"]
        items = sorted(self._shapes.items(), key=lambda kv: kv[1][0]*kv[1][1], reverse=True)
        mapping = {}
        for (name, _), key in zip(items, self.scale_keys):
            mapping[name] = key
        # 清理probe hook
        for h in self._probes:
            h.remove()
        self._probes.clear()
        self._mapping = mapping

    def register_injection(self, residuals: Dict[str, torch.Tensor]):
        def make_hook(name, key):
            def hook(module, inputs, output):
                R = residuals[key].to(dtype=output.dtype, device=output.device)

                # 尺寸匹配（空间分辨率）
                if R.shape[-2:] != output.shape[-2:]:
                    R = F.interpolate(R, size=output.shape[-2:], mode="bilinear", align_corners=False)

                # 通道匹配（输出通道数 vs R通道数）
                out_ch = output.shape[1]
                in_ch = R.shape[1]
                conv_key = f"{key}_{name}"
                if conv_key not in self._proj_convs:
                    # 延迟创建 1x1 卷积
                    self._proj_convs[conv_key] = nn.Conv2d(in_ch, out_ch, kernel_size=1).to(output.device, output.dtype)

                R = self._proj_convs[conv_key](R)
                return output + R  # 残差注入
            return hook

        for i, up in enumerate(self.unet.up_blocks):
            for j, res in enumerate(up.resnets):
                name = f"up{i}_res{j}"
                if name in getattr(self, "_mapping", {}):
                    key = self._mapping[name]
                    h = res.register_forward_hook(make_hook(name, key))
                    self._hooks.append(h)

    def clear(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._shapes.clear()
        
        
class TVRaftBackend(FlowBackend):
    def __init__(self, weights: Raft_Small_Weights = Raft_Small_Weights.DEFAULT, device=None):
        """
        RAFT-Small 光流后端（torchvision 版）。
        - 模型与预处理均来自 torchvision.models.optical_flow
        - 模型在初始化时设为 eval，并冻结梯度
        """
        super().__init__()
        self.model = raft_small(weights=weights)
        if device is not None:
            self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # torchvision 的 RAFT 预处理：将 [0,1] 归一化为 [-1,1] 等
        # 注意：当 weights 为 None 时没有 transforms，这里做个兜底
        self.transforms = weights.transforms() if hasattr(weights, "transforms") else None
        
        
    @torch.no_grad()
    def forward(self, lr_j: torch.Tensor, lr_i: torch.Tensor) -> torch.Tensor:
        """
        计算从邻帧 j 到目标帧 i 的光流（单位 = 原始 LR 像素位移）。
        自动处理 <128×128 时的上采样需求。
        """
        assert lr_j.ndim == 4, "Input must be [B,3,H,W]"
        B, C, H, W = lr_j.shape

        # ============ 1. 如果尺寸太小（<128），先放大到128，再算光流 ============
        need_resize = (H < 128 or W < 128)
        if need_resize:
            lr_j_in = F.interpolate(lr_j, size=(128, 128), mode="bilinear", align_corners=False)
            lr_i_in = F.interpolate(lr_i, size=(128, 128), mode="bilinear", align_corners=False)
        else:
            lr_j_in, lr_i_in = lr_j, lr_i

        # ============ 2. pad保证可以被8整除 ============
        B2, C2, H2, W2 = lr_j_in.shape
        pad_h = (8 - H2 % 8) % 8
        pad_w = (8 - W2 % 8) % 8
        if pad_h or pad_w:
            lr_j_pad = F.pad(lr_j_in, (0, pad_w, 0, pad_h), mode="replicate")
            lr_i_pad = F.pad(lr_i_in, (0, pad_w, 0, pad_h), mode="replicate")
        else:
            lr_j_pad, lr_i_pad = lr_j_in, lr_i_in

        # ============ 3. transforms（归一化） ============
        if self.transforms is not None:
            s_in, t_in = self.transforms(lr_j_pad, lr_i_pad)
        else:
            s_in = lr_j_pad * 2 - 1
            t_in = lr_i_pad * 2 - 1

        s_in = s_in.to(lr_j.device)
        t_in = t_in.to(lr_j.device)

        # ============ 4. RAFT 前向 ============ 
        try:
            outs = self.model(s_in, t_in, iters=12, test_mode=True)
        except TypeError:
            outs = self.model(s_in, t_in)

        flow = outs[-1] if isinstance(outs, (list, tuple)) else outs  # [B,2,H',W']

        # ============ 5. 如果放大过 → 缩回原LR大小 + 缩放光流幅度 ============
        if need_resize:
            # flow是基于128×128的像素位移，要缩回 H×W 尺寸，并调整幅值
            flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
            scale_x = float(W) / 128.0
            scale_y = float(H) / 128.0
            flow[:, 0] *= scale_x  # x方向
            flow[:, 1] *= scale_y  # y方向

        else:
            # 如果没放大，只需要cut出补边的部分
            if pad_h or pad_w:
                flow = flow[..., :H2, :W2]

        return flow.to(lr_j.dtype)



def grid_warp(img: torch.Tensor, flow_xy: torch.Tensor):
    """
    双线性采样：img 按 flow 像素位移（x,y）进行变形到 target 网格。
    img:  [B,C,H,W]  —— HR 或 LR 都可，但 flow 应与它同分辨率
    flow: [B,2,H,W] —— 像素位移，flow[:,0] 是 x（宽度方向），flow[:,1] 是 y（高度方向）
    """
    B, C, H, W = img.shape
    # 构建归一化光栅坐标
    dtype = img.dtype
    yy, xx = torch.meshgrid(
        torch.arange(H, device=img.device, dtype=dtype),
        torch.arange(W, device=img.device, dtype=dtype),
        indexing="ij",
    )
    # 像素坐标加位移 → 归一化到 [-1,1]
    x = (xx[None, ...] + flow_xy[:, 0, ...]) / max(W - 1, 1) * 2 - 1
    y = (yy[None, ...] + flow_xy[:, 1, ...]) / max(H - 1, 1) * 2 - 1
    grid = torch.stack([x, y], dim=-1)  # [B,H,W,2]
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)



class ReferenceBuilder:
    """
    解码-only + 光流 warp +（可选）QRM 打分。
    注意：全程 no_grad；像素不回传。
    """
    def __init__(self, vae_decode_fn, flow_estimator=None, use_qrm=False, device='cpu'):
        self.vae_decode = vae_decode_fn
        self.flow = flow_estimator
        self.use_qrm = use_qrm
        self.qrm = QRM(device=device) if use_qrm else None

    @torch.no_grad()
    def __call__(
        self,
        pred_x0_list: List[torch.Tensor],  # list of [B,4,h,w] (latent x0)
        lr_list: List[torch.Tensor],       # list of [B,3,H,W] (LR RGB, 0-1)
        center_idx: int = 0,
        topk: Optional[int] = None,
    ) -> torch.Tensor:
        # 1) 解码所有 x0 → HR RGB [0,1]
        hr_list = [self.vae_decode(z) for z in pred_x0_list]  # each [B,3,H,W_hr]
        B, _, H_hr, W_hr = hr_list[center_idx].shape

        # 2) 构造到 HR 的 flow：先在 LR 上估计，再上采到 HR，并做像素单位缩放
        tgt_lr = lr_list[center_idx]  # [B,3,H_lr,W_lr]
        H_lr, W_lr = tgt_lr.shape[-2:]
        scale_y = H_hr / max(H_lr, 1)
        scale_x = W_hr / max(W_lr, 1)

        warped, qs = [], []
        for k, (hrk, lrk) in enumerate(zip(hr_list, lr_list)):
            if k == center_idx:
                warped.append(hrk)
                qs.append(torch.ones(B, 1, H_hr, W_hr, device=hrk.device, dtype=hrk.dtype))
                continue

            flow_lr = self.flow(lrk, tgt_lr)  # [B,2,H_lr,W_lr], 像素位移（LR）
            flow_hr = F.interpolate(flow_lr, size=(H_hr, W_hr), mode="bilinear", align_corners=False)
            flow_hr[:, 0, ...] *= scale_x
            flow_hr[:, 1, ...] *= scale_y

            flow_hr = flow_hr.to(dtype=hrk.dtype, device=hrk.device)
            hrk_w = grid_warp(hrk, flow_hr)

            if self.use_qrm:
                lr_up = F.interpolate(tgt_lr, size=(H_hr, W_hr), mode="bilinear", align_corners=False)
                lr_up = lr_up.to(dtype=hrk_w.dtype, device=hrk_w.device)
                q = self.qrm(hrk_w, lr_up)
            else:
                q = torch.ones(B, 1, H_hr, W_hr, device=hrk.device, dtype=hrk.dtype)

            warped.append(hrk_w)
            qs.append(q)


        # 3) 选择 top-k 或直接拼接
        if topk is not None and self.use_qrm:
            means = torch.stack([q.mean(dim=[1, 2, 3]) for q in qs], dim=1)  # [B,K]
            idx = torch.topk(means, k=topk, dim=1).indices[0].tolist()
            parts = [(qs[i] * warped[i]) for i in idx]
            cond = torch.cat(parts, dim=1)  # [B,3*topk,H,W]
        else:
            cond = torch.cat([q * x for q, x in zip(qs, warped)], dim=1)  # [B,3K,H,W]
        return cond

# ========== SD×4 包装器 ==========
class SDX4_FRI_DSTCM_Wrapper(nn.Module):
    """
    将 FRI + DS-TCM 条件分支接到 SD×4 Upscaler 的 UNet 解码端。
    -嵌点：up_blocks.*.resnets.* 输出，做 residual add（Zero-Conv 初始化为 0）。
    -像素参考：decode-only + warp（不回传 VAE）。
    """
    def __init__(self, pipe, scale_factors=(8, 4, 2), use_qrm=False, K=3, device ='cpu', null_text_emb = None, logger = None, accelerator = None):
        super().__init__()
        self.pipe = pipe
        self.unet = pipe.unet.to(device)
        self.vae = pipe.vae.to(device)
        self.logger = logger
        self.accelerator = accelerator
        B = 1  # dummy batch size for null text embedding, only used during initialization
    
        self.null_encoder_hidden_states = null_text_emb  # SD-upscale 没 text encoder 也允许这样

        self.logger.info(f"null_encoder_hidden_states shape: {self.null_encoder_hidden_states.shape}", main_process_only=self.accelerator.is_main_process)

        self.flow_estimator = TVRaftBackend(device=next(self.unet.parameters()).device)

        
        # 以 factor 升序构造 key（面积从大到小）—— FIX: 确保与 resolve() 的排序一致
        scale_specs: Dict[str, ScaleSpec] = {}
        for f in sorted(scale_factors):  # e.g., (2,4,8)
            # 这里假定 up_blocks 的第 i 个对应 factor 的通道可取自 up_blocks[i] 的最后一个 resnet
            # 如果你的网络结构与 factor 顺序不同，请在此做更精确的映射。
            i = {2: 2, 4: 1, 8: 0}[f] if set(scale_factors) == {2, 4, 8} else list(scale_factors).index(f)
            ch = self.unet.up_blocks[i].resnets[-1].out_channels
            scale_specs[f"s{f}"] = ScaleSpec(ch=ch, factor=f)


        self.conditioner = FRI_DSTCM_Conditioner(n_rgbk=3 * K, scale_specs=scale_specs)

        
        self.snr_mlp = nn.Sequential(nn.Linear(1, 256), nn.SiLU(), nn.Linear(256, 256))

        def vae_decode_fn(latent_4chw):
            latent_4chw = latent_4chw.to(dtype=self.pipe.vae.dtype, device=next(self.unet.parameters()).device)
            out = self.vae.decode(
                latent_4chw / self.pipe.vae.config.scaling_factor, return_dict=False
            )[0]  # [-1,1]
            return (out.clamp(-1, 1) + 1) / 2.0

        self.ref_builder = ReferenceBuilder(vae_decode_fn, flow_estimator=self.flow_estimator, use_qrm=use_qrm, device=next(self.unet.parameters()).device)
        self.K = K
        
        # 关键：scale_keys 以 factor 升序（面积从大到小）传入
        self.injector = UNetMultiScaleInjector(self.unet, scale_keys=list(scale_specs.keys()))
        self.injector.probe()

        with torch.no_grad():
            device = next(self.unet.parameters()).device
            dtype  = next(self.unet.parameters()).dtype
            H = W = 64  # 任意 8 的倍数即可
            in_ch = getattr(self.unet.config, "in_channels", 7)
            dummy = torch.zeros(B, in_ch, H, W, device=device, dtype=dtype)
            self.logger.info(f"dummy shape, device, and dtype: {dummy.shape}, {dummy.device}, {dummy.dtype}", main_process_only=self.accelerator.is_main_process)
            # 兼容 upscaler：没有文本条件
            t0 = torch.zeros(B, dtype=torch.long, device=device)
            _ = self.unet(sample=dummy, 
                          timestep=t0, 
                          encoder_hidden_states=self.null_encoder_hidden_states[:B],
                          class_labels=None)
        self.injector.resolve()
        self._resolved = True



    @staticmethod
    def x0_from_eps(z_t: torch.Tensor, pred_eps: torch.Tensor, alpha_bar_t: torch.Tensor) -> torch.Tensor:
        """
        由 ε 预测反推 x0：
            z_t = sqrt(ā) * x0 + sqrt(1-ā) * ε
          => x0 = (z_t - sqrt(1-ā) * ε) / sqrt(ā)
        形状：
            z_t:        [B,4,h,w]
            pred_eps:   [B,4,h,w]
            alpha_bar_t:[B]
        """
        alpha_bar_t = alpha_bar_t.to(dtype=z_t.dtype)
        a_sqrt = alpha_bar_t.sqrt()[:, None, None, None]
        one_minus_a_sqrt = (1.0 - alpha_bar_t).sqrt()[:, None, None, None]
        return (z_t - one_minus_a_sqrt * pred_eps) / a_sqrt


    def build_and_register(
        self,
        z_t_list: List[torch.Tensor],         # [z_{j,t}] for j in N(i), 每个 [B,4,h,w]
        lr_rgb_seq: List[torch.Tensor],       # [lr_j] 同顺序
        pred_eps_list: List[torch.Tensor],    # [ε̂_{j,t}] 同顺序
        alpha_bar_t: torch.Tensor,            # [B]
        timestep: torch.Tensor,               # [B]
        center_idx: int,
    ):
        # 1) ε̂_{j,t} → x̂_{j,0}（latent）
        pred_x0_list = [ self.x0_from_eps(z_j_t, eps_j_t, alpha_bar_t)
                         for z_j_t, eps_j_t in zip(z_t_list, pred_eps_list) ]

        # 2) 解码 + LR 光流 j→i 放大/对齐，拼 cond_i
        with torch.no_grad():
            cond_i = self.ref_builder(pred_x0_list, lr_rgb_seq, center_idx=center_idx)

        # 3) DS-TCM 生成多尺度残差并注入
        # FIX: 统一 dtype
        t_emb  = timestep_embedding(timestep, dim=256).to(dtype=cond_i.dtype, device=cond_i.device)
        snr    = compute_snr(alpha_bar_t).unsqueeze(-1).to(dtype=cond_i.dtype, device=cond_i.device)
        snr_emb = self.snr_mlp(snr).to(dtype=cond_i.dtype, device=cond_i.device)        
        
        residuals = self.conditioner(cond_i, t_emb, snr_emb)

            
        self.injector.register_injection(residuals)


    def clear(self):
        self.injector.clear()

