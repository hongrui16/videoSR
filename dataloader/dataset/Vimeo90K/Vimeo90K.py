import os, glob, random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import torch.nn.functional as F


def build_vimeo90k_clips(sequences_root: str, split_txt: str) -> List[str]:
    """
    split_txt lines look like: 00001/0389
    We map to: <sequences_root>/00001/0389
    """
    clips = []
    with open(split_txt, "r") as f:
        for line in f:
            key = line.strip()
            if not key:
                continue
            clip_dir = os.path.join(sequences_root, key)
            
            # minimal existence check (im1.png)
            # if os.path.exists(os.path.join(clip_dir, "im1.png")):
            #     clips.append(clip_dir)
            
            clips.append(clip_dir)
            
    if not clips:
        raise RuntimeError(f"No clips found. Check paths. sequences_root={sequences_root}, split_txt={split_txt}")
    return clips

def _to_tensor_and_norm(img_np: np.ndarray, to_neg1_pos1: bool=False) -> torch.Tensor:
    # img_np: HxWx3 uint8
    img_np  = img_np.copy()
    t = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0
    if to_neg1_pos1:
        t = t * 2 - 1
    return t




class Vimeo90KNeighbor3Dataset(Dataset):
    def __init__(self, sequences_root = None, split = None, scale=4, crop_size=(256, 256), to_neg1_pos1=False, **kwargs):
                
        if sequences_root is None:
            sequences_root = "/scratch/rhong5/dataset/Vimeo90K/vimeo_septuplet/sequences"

        if split == "train":
            split_txt = "/scratch/rhong5/dataset/Vimeo90K/vimeo_septuplet/sep_trainlist.txt"
        elif split in ["val", "test"]:
            split_txt = "/scratch/rhong5/dataset/Vimeo90K/vimeo_septuplet/sep_testlist.txt"
        else:
            raise ValueError(f"Unknown split {split}. Provide split_txt.")
        
        self.scale = scale
        self.crop_size = crop_size
        self.to_neg1_pos1 = to_neg1_pos1
        
        self.clips = build_vimeo90k_clips(sequences_root, split_txt)
        self.debug = kwargs.get('debug', False)
        if self.debug:
            max_num = min(1500, len(self.clips))
            self.clips = self.clips[:max_num]
            
        


    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_dir = self.clips[idx]

        # 1. 随机选择中心帧 id ∈ [0, 6]
        center = random.randint(0, 6)

        # 2. 处理边界（mirror padding）
        if center == 0:
            ids = [0, 0, 1]
        elif center == 6:
            ids = [5, 6, 6]
        else:
            ids = [center - 1, center, center + 1]

        # 3. 读 3 张 HR 图
        hr_list = []
        for i in ids:
            path = os.path.join(clip_dir, f"im{i+1}.png")  # 文件名从1开始
            hr = np.array(Image.open(path).convert("RGB"))
            hr_list.append(hr)

        # 4. 随机裁剪 + 翻转增强
        if self.crop_size is not None:
            H, W = hr_list[0].shape[:2]
            ch, cw = self.crop_size
            top = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
            hr_list = [img[top:top+ch, left:left+cw] for img in hr_list]
            if random.random() < 0.5:
                hr_list = [img[:, ::-1] for img in hr_list]

        # 5. 转 Tensor
        HR = torch.stack([_to_tensor_and_norm(img, self.to_neg1_pos1) for img in hr_list], dim=0)  # [3,3,H,W]

        # 6. 生成 LR (下采样)
        H_hr, W_hr = HR.shape[-2:]
        LR = F.interpolate(HR, size=(H_hr // self.scale, W_hr // self.scale), mode="bicubic", align_corners=False)

        return {
            "lr_seq": LR,   # [3,3,h,w]
            "hr_seq": HR,   # [3,3,H,W]
        }


class Vimeo90KTestDataset(Dataset):
    def __init__(self, sequences_root, split, **kwargs):
        self.debug = kwargs.get('debug', False)
        
        self.clips = build_vimeo90k_clips(sequences_root, split)
        if self.debug:
            max_num = min(200, len(self.clips))
            self.clips = self.clips[:max_num]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip_dir = self.clips[idx]
        # 返回7帧路径，推理时外面用中心帧 t=0~6 枚举并处理边界
        frame_paths = [os.path.join(clip_dir, f"im{i}.png") for i in range(1, 8)]
        
        return {
                "lr_seq_path": frame_paths,  
                "hr_seq_path": frame_paths,  
            }
