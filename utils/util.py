from typing import List, Tuple, Union
import torch


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

def pick_neighbors_for_eval(seq, center, K = 3):
    """
    专用于 eval 的邻居采样：
    - seq: List[T] of tensors, each [B,3,H,W]
    - center: 指定中心帧 index（0 ≤ center < T）
    - K: 固定邻居帧数量（如 3, 5, 7）
    返回：
    - neighbors: List[K] of tensors
    - center_idx: int，中心帧在 neighbors 中的位置（始终为 K//2）
    """
    T = len(seq)
    half = K // 2

    neighbors = []
    for offset in range(-half, half + 1):
        idx = center + offset
        # 边界补齐策略：超出边界就复制最近的合法帧
        if idx < 0:
            idx = 0
        elif idx >= T:
            idx = T - 1
        neighbors.append(seq[idx])

    center_idx = half  # 中心永远落在 neighbors 的中间位置
    return neighbors, center_idx

