import os
import random
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def _to_tensor_and_norm(img_np: np.ndarray, to_neg1_pos1: bool) -> torch.Tensor:
    t = torch.from_numpy(img_np).float() / 255.0   # [H,W,3] in [0,1]
    t = t.permute(2, 0, 1).contiguous()            # [3,H,W]
    return t * 2 - 1 if to_neg1_pos1 else t

class REDSNeighbor3Dataset(Dataset):
    def __init__(self, root = None, split="train", scale=4, crop_size=256, to_neg1_pos1=False, **kwargs):
        if root is None:
            root = "/scratch/rhong5/dataset/REDS"
        self.hr_root = os.path.join(root, split, f"{split}_sharp")
        self.lr_root = os.path.join(root, split, f"{split}_sharp_bicubic", f"X{scale}")
        self.video_list = sorted(os.listdir(self.hr_root))
        self.crop_size = crop_size
        self.to_neg1_pos1 = to_neg1_pos1
        self.debug = kwargs.get('debug', False)
        if self.debug:
            max_num = min(1500, len(self.video_list))
            self.video_list = self.video_list[:max_num]

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        seq = self.video_list[index]
        hr_dir = os.path.join(self.hr_root, seq)
        lr_dir = os.path.join(self.lr_root, seq)

        # 随机选中心帧 [0, 99]
        center = random.randint(0, 99)

        # ✅ 边界复制，避免 t-1 / t+1 越界
        if center == 0:
            ids = [0, 0, 1]
        elif center == 99:
            ids = [98, 99, 99]
        else:
            ids = [center - 1, center, center + 1]

        hr_list, lr_list = [], []
        for i in ids:
            hr = np.array(Image.open(os.path.join(hr_dir, f"{i:08d}.png")).convert("RGB"))
            lr = np.array(Image.open(os.path.join(lr_dir, f"{i:08d}.png")).convert("RGB"))
            hr_list.append(hr)
            lr_list.append(lr)

        # 随机裁剪（保证HR 和 LR 对齐）
        if self.crop_size is not None:
            H, W = hr_list[0].shape[:2]
            x = random.randint(0, H - self.crop_size)
            y = random.randint(0, W - self.crop_size)
            hr_list = [img[x:x+self.crop_size, y:y+self.crop_size] for img in hr_list]
            lr_list = [img[x//4:(x+self.crop_size)//4, y//4:(y+self.crop_size)//4] for img in lr_list]

        HR = torch.stack([_to_tensor_and_norm(img, self.to_neg1_pos1) for img in hr_list])  # [3,3,H,W]
        LR = torch.stack([_to_tensor_and_norm(img, self.to_neg1_pos1) for img in lr_list])  # [3,3,h,w]

        return {
            "lr_seq": LR,   # [t,3,h,w]
            "hr_seq": HR,   # [t,3,H,W]
        }


class REDSTestVideoDataset(Dataset):
    def __init__(self, root, split="val", scale=4, **kwargs):
        self.debug = kwargs.get('debug', False)
        split = 'val' if split == 'test' else split
        self.hr_root = os.path.join(root, split, f"{split}_sharp")
        self.lr_root = os.path.join(root, split, f"{split}_sharp_bicubic", f"X{scale}")
        self.video_list = sorted(os.listdir(self.hr_root))

        if self.debug:
            max_num = min(200, len(self.video_list))
            self.video_list = self.video_list[:max_num]

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):
        seq = self.video_list[index]
        hr_dir = os.path.join(self.hr_root, seq)
        lr_dir = os.path.join(self.lr_root, seq)

        hr_frames = sorted(glob.glob(os.path.join(hr_dir, "*.png")))
        lr_frames = sorted(glob.glob(os.path.join(lr_dir, "*.png")))

        # 只返回路径，后续 eval_step_per_sequence 再读取和处理
        return {
            "lr_seq_path": lr_frames,  
            "hr_seq_path": hr_frames,  
        }

