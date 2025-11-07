from typing import List, Tuple, Union
import torch
import os
import numpy as np
from PIL import Image
import torch.nn.functional as F 

def split_seq(x: Union[List[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
    if isinstance(x, list):
        return x
    # x: [B,T,3,H,W]
    return [x[:, t] for t in range(x.shape[1])]

def pick_neighbors_fixed(
    seq: List[torch.Tensor], 
    K: int, 
    center_mode="random"
) -> Tuple[List[torch.Tensor], int]:
    """
    返回始终为 K 帧的 neighbors；若靠近边界，则自动补齐（复制边界帧）。
    seq: list of T tensors, each (B,3,H,W)
    return: neighbors (len = K), center_idx (在 neighbors 中的位置)
    """
    T = len(seq)

    # 1. 选择中心帧 index c
    if center_mode == "fixed0":
        c = 0
    elif center_mode == "random":
        c = torch.randint(0, T, (1,)).item()
    else:
        c = T // 2

    # 2. 初始化邻居帧
    idxs = [c]
    l = r = 1

    while len(idxs) < K and (c - l >= 0 or c + r < T):
        # 尽可能向左取
        if c - l >= 0:
            idxs.append(c - l)
            l += 1
        # 再尽可能向右取
        if len(idxs) < K and c + r < T:
            idxs.append(c + r)
            r += 1

    # 3. 如果还不足 K 帧 → 用边界帧复制补齐
    if len(idxs) < K:
        while len(idxs) < K:
            # 复制中心帧，或者复制最近的帧都可以
            idxs.append(c)

    # 4. 排序，保持时间顺序
    idxs = sorted(idxs)

    neighbors = [seq[i] for i in idxs]
    center_idx = idxs.index(c)

    return neighbors, center_idx


def _to_tensor(img_path, to_neg1_pos1=False, scale =1):
    img = np.array(Image.open(img_path).convert("RGB"))
    t = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    if scale !=1:
        C, H, W = t.shape
        t = F.interpolate(t.unsqueeze(0), size=(H//scale, W//scale), mode="bicubic", align_corners=False).squeeze(0)
    return t * 2 - 1 if to_neg1_pos1 else t

def pick_neighbors_for_eval(seq_paths, center, device, to_neg1_pos1=False, scale=1):
    """
    seq_paths: List[List[str]]  # 形状是 [B, T]，每个元素是图片路径，例如:
                                # [["path/000/im1.png", ..., "im7.png"],  # 第一个视频
                                #  ["path/001/im1.png", ..., "im7.png"]]  # 第二个视频
    center:      当前中心帧 (0-based)
    dataset_name: "REDS" or "Vimeo90K"
    返回:
        batch frames_3: Tensor [B, 3, 3, H, W]
    """
    batch_tensors = []
    for frame_list in seq_paths:  # frame_list: List[str] 长度 T
        T = len(frame_list)
        if T == 0:
            raise ValueError("Empty frame list in seq_paths.")
        if T == 1:
            ids = [0, 0, 0]
        else:
            if center <= 0:
                ids = [0, 0, 1]
            elif center >= T - 1:
                ids = [T - 2, T - 1, T - 1]
            else:
                ids = [center - 1, center, center + 1]

        imgs = [_to_tensor(frame_list[i], to_neg1_pos1, scale) for i in ids]  # 3 × [3,H,W]

        # 可选：确保同一 batch 内尺寸一致（如果你预先裁剪/对齐了就不需要）
        # 这里假设序列内各帧同尺寸，直接堆叠：
        clip = torch.stack(imgs, dim=0)  # [3,3,H,W]
        batch_tensors.append(clip)

    return torch.stack(batch_tensors, dim=0).to(device)  # [B,3,3,H,W]
