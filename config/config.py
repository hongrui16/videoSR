
import os
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional


# =======================
#        CONFIG
# =======================
@dataclass
class Config:
    seed: int = 42
    dataset_name: str = "Vimeo-90K"
    train_bsz: int = 20
    num_workers: int = 4
    lr: float = 1e-4
    max_epochs: int = 100
    log_every: int = 300
    eval_every: int = 5000
    save_every: int = 2000
    ckpt_out_dir: str =  '/scratch/rhong5/weights/temp_training_weights'
    log_dir: str = 'zlog'
    resume_from: str = "/scratch/rhong5/weights/temp_training_weights/sdx4_fri_dstcm/Vimeo-90K/20251103_110405_5000220/fri_dstcm.pt"  # e.g., ./ckpts_sdx4_fri_dstcm/fri_dstcm_step_10000.pt
    runtime_mode: str = "train"  # "train" | "eval"
    use_qrm: bool = False
    use_band_consistency: bool = True
    band_weight: float = 0.1
    K_neighbors: int = 3            # e.g. {t-1,t,t+1}
    grad_accum: int = 1
    center_idx_mode: str = "random" # "middle" | "random" | "fixed0"
    network_type: str = "sdx4_fri_dstcm"
    # eval
    num_eval_batches: int = 6
    num_inference_steps_eval: int = 50
    finetune: bool = False    
    dump_sr: bool = False
    # model_weights_precision: str = "fp32"  # "fp16" | "bf16" | "fp32"
    unet_weights_precision: str = "fp32"  # "fp16" | "bf16" | "fp32"
    data_precision: str = "fp32"  # "fp16" | "bf16" | "fp32"
    mixed_precision: str = "no"   # "no" | "fp16" | "bf16"
    

