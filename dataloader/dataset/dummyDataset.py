import torch
import random

class DummyDataset(torch.utils.data.Dataset):
    """
    Yields tuples (lr_frames, hd_frames, sparse_indices) with shapes:
      lr_frames:  [1, T, 3, H, W]
      hd_frames:  [1, N, 3, H, W]  (N in {0,1,2})
      sparse_indices: [1, N] (long)
    """
    def __init__(self, T=14, H=256, W=256):
        self.nums = 2000
        self.T, self.H, self.W = T, H, W

        
        
        
        self.height = H
        self.width = W

    def __len__(self):
        return self.nums

    def __getitem__(self, idx):
        """
        Yields:
        lr_frames:    [T, C, H, W], 值域 [-1, 1]
        hd_frames:    [2, C, H, W]  固定两个候选槽位（第0帧 & T//2-1）
        mask:         [2]           逐样本启用的槽位（0/1/2）
        """
        C, T, H, W =3, self.T, self.H, self.W
        assert T >= 2, "T must be >= 2 to use indices [0, T//2 - 1 ]"

        # LR video in [0, 1]
        lr_frames = torch.rand(T, 3, H//4, W//4)

        hd_frames = torch.rand(T, 3, H, W)
        
        batch = {
            "lr_seq": lr_frames,    # [1,T,3,H,W]
            "hr_seq": hd_frames,    # [1,T,3,H,W]
        }


        return batch


if __name__ == "__main__":
    dataset = DummyDataset(device="cpu")
    for i in range(10):
        lr_frames, hd_frames, mask = dataset[i]
        print(f"Sample {i}:")
        print("  lr_frames:", lr_frames.shape, lr_frames.min().item(), lr_frames.max().item())
        print("  hd_frames:", hd_frames.shape, hd_frames.min().item(), hd_frames.max().item())
        print("  mask:", mask, mask.sum().item())