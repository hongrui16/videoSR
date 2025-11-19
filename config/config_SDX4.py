
import os
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional


# =======================
#        CONFIG
# =======================
@dataclass
class Config:
    seed: int = 42
    train_bsz: int = 20
    num_workers: int = 4
    lr: float = 1e-4
    max_epochs: int = 40
    log_every: int = 300
    eval_every: int = 5000
    save_every: int = 2000
    ckpt_out_dir: str =  '/scratch/rhong5/weights/temp_training_weights'
    log_dir: str = 'zlog'
    resume_from: str = ""  # e.g., ./ckpts_sdx4_fri_dstcm/fri_dstcm_step_10000.pt    
    # dataset_name: str = "Vimeo-90K"
    # resume_from: str = "/scratch/rhong5/weights/temp_training_weights/sdx4_fri_dstcm/Vimeo-90K/20251109_025032_5012784/fri_dstcm_best.pt"  # e.g., ./ckpts_sdx4_fri_dstcm/fri_dstcm_step_10000.pt
    dataset_name: str = "REDS"
    resume_from: str = "/scratch/rhong5/weights/temp_training_weights/SD-X4/REDS/2025-11-18-22-49-16_5036959/SD-X4_best.pt"  # e.g., ./ckpts_sdx4_fri_dstcm/fri_dstcm_step_10000.pt
    runtime_mode: str = "train"  # "train" | "eval"
    use_qrm: bool = False
    use_band_consistency: bool = True
    band_weight: float = 8
    use_rec_loss: bool = True
    rec_weight: float = 2  # 建议 1~1.5
    K_neighbors: int = 3            # e.g. {t-1,t,t+1}
    grad_accum: int = 1
    center_idx_mode: str = "random" # "middle" | "random" | "fixed0"
    network_type: str = "SD-X4"
    # eval
    num_eval_batches: int = 6
    num_inference_steps_eval: int = 50
    finetune: bool = False    
    dump_sr: bool = False
    unet_weights_precision: str = "fp32"  # "fp16" | "bf16" | "fp32"
    data_precision: str = "fp32"  # "fp16" | "bf16" | "fp32"
    mixed_precision: str = "no"   # "no" | "fp16" | "bf16"
    use_lr_interp_for_ref: bool = True
    use_lora: bool = True
    use_injection: bool = False
    

