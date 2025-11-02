import torch
import torch.nn as nn
import torch.nn.functional as F


# ========== Laplacian Pyramid & tiny encoders ==========
class LapPyramid(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        k = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
        k = (k[:, None] * k[None, :])
        k = k / k.sum()
        self.register_buffer("kernel", k[None, None, ...])  # [1,1,5,5]
        self.channels = channels

    def gaussian(self, x):
        k = self.kernel.repeat(self.channels, 1, 1, 1).to(dtype=x.dtype)
        return F.conv2d(x, k, padding=2, groups=self.channels)


    def forward(self, x: torch.Tensor):
        low1 = self.gaussian(x)
        high = x - low1
        low2 = self.gaussian(low1)
        mid = low1 - low2
        low = low2
        return low, mid, high


# ========== 低分辨率带通一致性（stop-grad） ==========
class BandConsistencyLoss(nn.Module):
    def __init__(self, weight_low=0.5, weight_mid=0.7, weight_high=1.0, down=2):
        super().__init__()
        self.w = {"low": weight_low, "mid": weight_mid, "high": weight_high}
        self.down = down
        self.lap = LapPyramid(channels=3)

    @torch.no_grad()
    def _bands(self, img):
        if self.down > 1:
            img = F.interpolate(img, scale_factor=1.0 / self.down, mode="bilinear", align_corners=False)
        return self.lap(img)

    def forward(self, I_pred: torch.Tensor, I_ref: torch.Tensor):
        with torch.no_grad():
            l_ref, m_ref, h_ref = self._bands(I_ref)
        l_p, m_p, h_p = self._bands(I_pred)
        loss = (
            self.w["low"] * F.smooth_l1_loss(l_p, l_ref)
            + self.w["mid"] * F.smooth_l1_loss(m_p, m_ref)
            + self.w["high"] * F.smooth_l1_loss(h_p, h_ref)
        )
        return loss
