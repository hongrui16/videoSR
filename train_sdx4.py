# train_sdx4_fri_dstcm_full.py (refactored)
# SD×4 + FRI + DS-TCM —— diffusion ε-target
# - UNet/VAE frozen; train conditioner (+optional QRM)
# - Supports dataloader: list[T * (B,3,H,W)] or tensor (B,T,3,H,W)
# - Single cohesive Trainer class; no duplicated functions

import os
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
import shutil
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import StableDiffusionUpscalePipeline, EulerAncestralDiscreteScheduler
import torch.nn as nn
import argparse
from datetime import datetime
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils import DistributedDataParallelKwargs
import logging
from accelerate.logging import get_logger
from contextlib import nullcontext
import numpy as np
import random
import PIL.Image as Image
import diffusers
from diffusers.models.attention_processor import LoRAAttnProcessor
from peft import LoraConfig, inject_adapter_in_model
from peft import get_peft_model_state_dict, set_peft_model_state_dict

import inspect

# from config.config import TrainConfig
from dataloader.build_dataloader import build_dataloader
from cond_modules import SDX4_FRI_DSTCM_Wrapper
from utils.loss import BandConsistencyLoss
from utils.util import split_seq, pick_neighbors_for_eval, img_list_to_tensor, plot_loss_curves
from utils.lora import enable_lora, verify_lora_injected, verify_lora_gradients, log_module_info
from utils.scheduler import construct_noisy_latent_and_target, sample_timesteps, x0_from_eps, x0_from_vpred

LOGGING_DIR = os.getenv("LOG_DIR", None)
OUTPUT_CKPTS_DIR = os.getenv("WEIGHT_DIR", None)

if LOGGING_DIR is not None:
    from config import Config
else:
    from config.config_SDX4 import Config


print("Diffusers version:", diffusers.__version__)
print("LoRAAttnProcessor import path:", LoRAAttnProcessor.__module__)
print("LoRAAttnProcessor source:", inspect.getsource(LoRAAttnProcessor))


def get_autocast_context(accelerator: Accelerator):
    if accelerator.mixed_precision == "fp16":
        return torch.autocast(device_type=accelerator.device.type, dtype=torch.float16)
    elif accelerator.mixed_precision == "bf16":
        return torch.autocast(device_type=accelerator.device.type, dtype=torch.bfloat16)
    else:  # "no"
        return nullcontext()


# =======================
#  PIPE & VAE HELPERS
# =======================

def _must_finite(x, tag):
    if not torch.isfinite(x).all():
        raise RuntimeError(f"[NaNCheck] {tag}: found non-finite values "
                           f"(min={x.min().item()}, max={x.max().item()})")


def prepare_pipe(unet_weights_precision, device, logger=None):
    if unet_weights_precision == "fp32":
        weight_dtype = torch.float32
    elif unet_weights_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float16  # 默认 fp16

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=weight_dtype,
    )
        
    
    pipe.unet.config.num_class_embeds = None          # 1) 修改 config
    if hasattr(pipe.unet, "class_embedding"):         # 2) 删除类嵌入模块
        pipe.unet.class_embedding = None
    pipe.unet.get_class_embed = lambda *args, **kwargs: None  # 3) 避免 get_class_embed 里再报错

    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.eval()
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder.to(device)     
        
    pipe.to(device=device, dtype=weight_dtype)
    pipe.unet.eval()
    pipe.unet.requires_grad_(False)
        
    pipe.vae.to(device=device, dtype=torch.float32)
    pipe.vae.eval()
    pipe.vae.requires_grad_(False)

    if logger is not None:
        logger.info(f"Scheduler: {pipe.scheduler.__class__.__name__}", main_process_only=True) #DDIMScheduler
        logger.info(f"Scheduler prediction_type: {pipe.scheduler.config.prediction_type}", main_process_only=True)
        logger.info(f"Pipeline prepared. pipe unet dtype: {next(pipe.unet.parameters()).dtype}", main_process_only=True)
        logger.info(f"Pipeline prepared. pipe vae dtype: {next(pipe.vae.parameters()).dtype}", main_process_only=True)
        logger.info(f"Pipeline prepared. pipe text_encoder dtype: {next(pipe.text_encoder.parameters()).dtype}", main_process_only=True)
    else:
        print("Scheduler:", pipe.scheduler.__class__.__name__)        #DDIMScheduler
        print("Scheduler prediction_type:", pipe.scheduler.config.prediction_type) #v_prediction
        print(f"Pipeline prepared. pipe unet dtype: {next(pipe.unet.parameters()).dtype}")  
        print(f"Pipeline prepared. pipe vae dtype: {next(pipe.vae.parameters()).dtype}")
        print(f"Pipeline prepared. pipe text_encoder dtype: {next(pipe.text_encoder.parameters()).dtype}")        
    return pipe

@torch.no_grad()
def vae_encode_img(pipe, img_01: torch.Tensor) -> torch.Tensor:
    """
    Safe encoding with stable FP32 compute, without permanently changing VAE dtype.
    """
    # 1) 输入从 [0,1] -> [-1,1]，并转为 float32
    img = (img_01 * 2 - 1).to(dtype=torch.float32)

    # 2) 做 encode（FP32下很稳定，不会NaN）
    posterior = pipe.vae.encode(img).latent_dist
    z = posterior.mean * pipe.vae.config.scaling_factor

    return z.to(dtype=torch.float32)  # 输出保持 float32，后面更安全



def vae_decode_latent(pipe, z_4chw: torch.Tensor) -> torch.Tensor:
    """ latent [B,4,h,w] -> img_01 in [0,1], shape [B,3,H,W] """
    z_4chw = z_4chw.to(dtype=pipe.vae.dtype)
    out = pipe.vae.decode(z_4chw / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return (out.clamp(-1, 1) + 1) / 2.0




class SD_X4_worker:
    def __init__(self, args: argparse.Namespace):        
        cfg = Config()        
        self.args = args        
        self.debug = args.debug        
        self.runtime_mode = args.runtime if args.runtime else cfg.runtime_mode
        self.max_epochs = cfg.max_epochs
        self.finetune = cfg.finetune 
        self.dump_sr = args.dump_sr if args.dump_sr else cfg.dump_sr
        cfg.resume_from = args.resume if args.resume else cfg.resume_from
        use_lr_interp_for_ref = cfg.use_lr_interp_for_ref
        self.cfg = cfg        
        
        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # job_id = str(os.getpid())
        slurm_job_id = os.getenv('SLURM_JOB_ID')
        self.network_name = cfg.network_type
        dataset_name = args.dataset_name if args.dataset_name else cfg.dataset_name
        unet_weights_precision = args.unet_weights_precision if args.unet_weights_precision else cfg.unet_weights_precision
                
        if self.runtime_mode == 'train':
            if LOGGING_DIR is None:
                if self.debug:
                    self.logging_dir = os.path.join(cfg.log_dir, self.network_name, dataset_name, 'debug_' + current_timestamp + '_' + slurm_job_id)
                else:
                    self.logging_dir = os.path.join(cfg.log_dir, self.network_name, dataset_name, current_timestamp + '_' + slurm_job_id)
                # self.output_vis_dir = os.path.join(self.logging_dir, "vis")                    
                if self.debug:
                    self.ckpt_out_dir = self.logging_dir
                else:
                    self.ckpt_out_dir = os.path.join(cfg.ckpt_out_dir, self.network_name, dataset_name, current_timestamp + '_' + slurm_job_id)
            else:
                self.logging_dir = LOGGING_DIR
                self.ckpt_out_dir = OUTPUT_CKPTS_DIR
        else:  # eval
            if cfg.resume_from:
                parent_log_dir = os.path.dirname(cfg.resume_from).replace(cfg.ckpt_out_dir, cfg.log_dir)                 
                self.logging_dir = os.path.join(parent_log_dir, f'{current_timestamp}_{slurm_job_id}') 
            self.ckpt_out_dir = None  # not used in eval
            
        debug_dir = os.path.join(self.logging_dir, "zdebug")
        
        accelerator_project_config = ProjectConfiguration(project_dir=self.logging_dir, logging_dir=self.logging_dir)
        
        self.accelerator = Accelerator(mixed_precision=self.cfg.mixed_precision, 
                                       gradient_accumulation_steps=self.cfg.grad_accum,
                                        project_config=accelerator_project_config,
                                        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)] )
        self.accelerator.wait_for_everyone()
        
        self.autocast_context = get_autocast_context(self.accelerator)

        if self.accelerator.is_main_process:
            os.makedirs(self.logging_dir, exist_ok=True)
            if self.runtime_mode == 'train':
                os.makedirs(self.ckpt_out_dir, exist_ok=True)
                
        log_dir_last_name = os.path.basename(os.path.normpath(self.logging_dir))

        if self.dump_sr and self.accelerator.is_main_process:
            self.dump_dir = os.path.join(self.logging_dir, "sr_videos")
            os.makedirs(self.dump_dir, exist_ok=True)
                
        log_path = os.path.join(self.logging_dir, f"{log_dir_last_name}_info.log")
        ## copy current script to logging dir

        if self.accelerator.is_main_process and self.runtime_mode == 'train':
            code_backup_dir = os.path.join(self.logging_dir, f"code_backup_{slurm_job_id}")
            os.makedirs(code_backup_dir, exist_ok=True)
            shutil.copyfile('train_sdx4.py', os.path.join(code_backup_dir, f'train_sdx4_{slurm_job_id}.py'))
            # shutil.copyfile('cond_modules.py', os.path.join(code_backup_dir, f'cond_modules_{slurm_job_id}.py'))
            ## copy config dir to code_backup dir
            shutil.copytree('config', os.path.join(code_backup_dir, 'config'))
            shutil.copytree('dataloader', os.path.join(code_backup_dir, 'dataloader'))
            shutil.copytree('utils', os.path.join(code_backup_dir, 'utils'))

        handlers = []
        if self.accelerator.is_main_process:
            # 主进程记录到文件和控制台
            handlers = [logging.FileHandler(log_path), logging.StreamHandler()]
        else:
            # 非主进程只记录到控制台或空 handler
            handlers = [logging.StreamHandler()]

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            handlers=handlers
        )
        self.logger = get_logger(__name__)
                
        self.logger.info(f'logging_dir: \n{self.logging_dir}\n', main_process_only=self.accelerator.is_main_process)
        self.logger.info(f'ckpt_out_dir: \n{self.ckpt_out_dir}\n', main_process_only=self.accelerator.is_main_process)
        self.logger.info(f'mixed_precision: {self.cfg.mixed_precision}', main_process_only=self.accelerator.is_main_process)
        self.logger.info(f"World size: {self.accelerator.num_processes}", main_process_only=self.accelerator.is_main_process)

        self.logger.info(f"runtime_mode: {self.runtime_mode}", main_process_only=self.accelerator.is_main_process)
        self.logger.info(f"dataset_name: {dataset_name}", main_process_only=self.accelerator.is_main_process)
        self.logger.info(f"device: {self.accelerator.device}", main_process_only=self.accelerator.is_main_process)
        self.logger.info(f"unet_weights_precision: {unet_weights_precision}", main_process_only=self.accelerator.is_main_process)
        self.logger.info(f'dump_sr: {self.dump_sr}', main_process_only=self.accelerator.is_main_process)
        
        
        device = self.accelerator.device
                        
        self.pipe = prepare_pipe(unet_weights_precision, device, logger=self.logger)

        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = getattr(self.pipe, "text_encoder", None)
        self.text_encoder.to(device) if self.text_encoder is not None else None

        if self.cfg.use_lora:
            self.pipe.unet = enable_lora(self.pipe.unet, r=8, lora_alpha=16)
            verify_lora_injected(self.pipe.unet)
            verify_lora_gradients(self.pipe.unet)
                
        train_bsz = self.cfg.train_bsz    
        if self.text_encoder is not None:
            with torch.no_grad():
                tokens = self.tokenizer(
                    [""] * train_bsz,  # 空文本
                    max_length=self.tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                ).input_ids
                tokens = tokens.to(device)
                # 计算空文本 embedding
                self.null_text_emb = self.text_encoder(tokens)[0]  # [B, 77, hidden_dim]
        else:
            self.null_text_emb = None
            
        self.logger.info(f"null_text_emb device and shape: {self.null_text_emb.device if self.null_text_emb is not None else None}, {self.null_text_emb.shape if self.null_text_emb is not None else None}", main_process_only=self.accelerator.is_main_process)


        if self.debug:
            self.cfg.train_bsz = 2

        if self.runtime_mode == 'train':
            self.band_reg = BandConsistencyLoss().to(device) if self.cfg.use_band_consistency else None
            params = []
            for n, p in self.pipe.unet.named_parameters():
                if p.requires_grad:
                    params.append(p)
                    

                    
            self.optimizer = torch.optim.AdamW(params, lr=self.cfg.lr)                    
            self.train_loader, self.train_dataset = build_dataloader(
                split="train",
                batch_size=self.cfg.train_bsz,
                num_workers=self.cfg.num_workers,
                dataset_name=dataset_name,
                device=device,
                debug=self.debug,
            )                    
            self.logger.info(f"Train dataset size: {len(self.train_dataset)}", main_process_only=self.accelerator.is_main_process)      
            self.val_loader, self.val_dataset = build_dataloader(
                split="val",
                batch_size=self.cfg.train_bsz,
                num_workers=self.cfg.num_workers,
                dataset_name=dataset_name,
                device=device,
                debug=self.debug,

            )
            self.logger.info(f"Validation dataset size: {len(self.val_dataset)}", main_process_only=self.accelerator.is_main_process)
            self.optimizer, self.train_loader = self.accelerator.prepare(self.optimizer, self.train_loader)
            self.val_loader = self.accelerator.prepare(self.val_loader)
        else:                
            self.test_loader, self.test_dataset = build_dataloader(
                split="test",
                batch_size=self.cfg.train_bsz,
                num_workers=self.cfg.num_workers,
                dataset_name=dataset_name,
                device=device,
                debug=self.debug,
            )
            self.logger.info(f"Test dataset size: {len(self.test_dataset)}", main_process_only=self.accelerator.is_main_process)
                    
            if self.test_loader is not None:
                (self.test_loader) = self.accelerator.prepare(self.test_loader)


        
        

        self.logger.info(f"Data loaders prepared.", main_process_only=self.accelerator.is_main_process)
        


        # optional resume
        self.start_epoch = 0
        self.best_metric = float("-inf")
        self.best_loss = float("inf")
        if self.cfg.resume_from:
            with self.accelerator.main_process_first():
                self.start_epoch = self.load_checkpoint(self.cfg.resume_from, map_location=self.accelerator.device) + 1
            self.accelerator.wait_for_everyone()
            self.logger.info(f"Resumed from {self.cfg.resume_from} at epoch {self.start_epoch}", main_process_only=self.accelerator.is_main_process)
                    
    # ---------- checkpoint ----------
    def save_checkpoint(self, epoch, is_best = False, psnr : Optional[float] = None, rec_loss : Optional[float] = None):
        if not self.accelerator.is_main_process:
            return
        os.makedirs(self.ckpt_out_dir, exist_ok=True)
        # tag = tag or f"step_{self.step}"
        if is_best:
            path = os.path.join(self.ckpt_out_dir, f"{self.network_name}_best.pt")
        else:
            path = os.path.join(self.ckpt_out_dir, f"{self.network_name}.pt")

        if self.cfg.use_lora:
            unet_lora_state = get_peft_model_state_dict(self.pipe.unet)
        else:
            unet_lora_state = None


        ckpt = {
            "unet_lora": unet_lora_state,
            "optimizer": self.optimizer.state_dict(),  # accelerate 处理过也能保存
            "epoch": epoch,
            "psnr": psnr,
            "rec_loss": rec_loss,
        }

        
        torch.save(ckpt, path)
        # 统计 LoRA 参数量
        if self.cfg.use_lora:
            lora_params = sum(p.numel() for p in unet_lora_state.values())
            self.logger.info(
                f"[CKPT] Saved -> {path}\n"
                f"       LoRA params: {lora_params:,} (~{lora_params*4/1e6:.1f}MB)"
            )


    def load_checkpoint(self, ckpt_path: str, map_location: str = "cpu") -> int:
        if not os.path.isfile(ckpt_path):
            self.logger.warning(f"[CKPT] No checkpoint found at {ckpt_path}")
            return -1
        
        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.logger.info(f"[CKPT] loaded <- {ckpt_path}")

        # 恢复 UNet 的 LoRA 权重
        # 用 PEFT 加载 LoRA（自动匹配模块）
        if self.cfg.use_lora and "unet_lora" in ckpt and ckpt["unet_lora"]:
            set_peft_model_state_dict(self.pipe.unet, ckpt["unet_lora"])
            
            # 验证加载成功
            loaded_params = sum(p.numel() for p in ckpt["unet_lora"].values())
            self.logger.info(f"[CKPT] LoRA loaded: {loaded_params:,} params", main_process_only=self.accelerator.is_main_process)


            

        if self.finetune or self.runtime_mode != 'train':
            return -1
        
        # 载入优化器（若在训练中恢复）
        if self.optimizer is not None and ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.logger.info(f"[CKPT] optimizer loaded <- {ckpt_path}")

        self.best_metric = ckpt.get("psnr", float("-inf"))
        if self.best_metric is None:
            self.best_metric = float("-inf")
        self.best_loss = ckpt.get("rec_loss", float("inf"))
        if self.best_loss is None:
            self.best_loss = float("inf")
        self.logger.info(f"[CKPT] best_metric loaded <- {ckpt_path}")
            
        return int(ckpt.get("epoch", 0))

    def training_validation_step(
        self,
        t_int: torch.Tensor,
        a_bar: torch.Tensor,
        lr_seq_neighbors: List[torch.Tensor],
        hr_seq_neighbors: List[torch.Tensor],
        center_idx: int,
        center_z_t: torch.Tensor,
        is_train: bool = True,
        ):
        """
        完整时序 teacher-forcing：
        1. 对每个邻帧 j 构造 z_{j,t} 并预测 ε̂_{j,t}
        2. warp(j→i) 得到 cond_i
        3. 注入 UNet 解码端
        4. 预测中心帧 i 的 ε̂_{i,t}
        """
        device = self.accelerator.device
        B = lr_seq_neighbors[0].shape[0]

        # ===== (1) Teacher-forcing 所有邻帧 j =====
        with self.autocast_context:
            z_i_t = center_z_t               
            z_i_t_scaled = self.pipe.scheduler.scale_model_input(z_i_t.to(self.pipe.unet.dtype), t_int[0])

            lr_i = lr_seq_neighbors[center_idx]
            lr_i_norm = lr_i * 2 - 1
            lr_i_lat = F.interpolate(lr_i_norm, size=z_i_t.shape[-2:], mode="bilinear", align_corners=False)

            # z_i_7ch = torch.cat([z_i_t, lr_i_lat], dim=1)
            z_i_7ch = torch.cat([z_i_t_scaled, lr_i_lat], dim=1)
            _must_finite(z_i_7ch, "z_i_7ch")
            
            z_i_7ch = z_i_7ch.to(dtype=self.pipe.unet.dtype)
            pred_out_i = self.pipe.unet(sample=z_i_7ch, 
                                        timestep=t_int, 
                                        encoder_hidden_states=self.null_text_emb[:z_i_7ch.shape[0]], 
                                        class_labels=None).sample

        return pred_out_i


    # ---------- eval ----------
    @torch.no_grad()
    def _psnr(self, x: torch.Tensor, y: torch.Tensor) -> float:
        mse = F.mse_loss(x, y, reduction="mean").item()
        if mse == 0: return 100.0
        return 10.0 * torch.log10(torch.tensor(1.0) / torch.tensor(mse)).item()


    def training_validation_epoch(self, data_loader, epoch, is_train = True, use_rec_loss = False, use_band_consistency_loss = False):
        if is_train:
            mode = "train"
            torch.set_grad_enabled(True)
            return self._run_epoch(mode, data_loader, epoch, is_train=True, use_rec_loss=use_rec_loss, use_band_consistency_loss=use_band_consistency_loss) 
        else:
            mode = "val"            
            with torch.no_grad():
                return self._run_epoch(mode, data_loader, epoch, is_train=False, use_rec_loss=use_rec_loss, use_band_consistency_loss=use_band_consistency_loss)

    def _run_epoch(self, mode, data_loader, epoch, is_train = True, use_rec_loss = False, use_band_consistency_loss = False):
        total_loss = 0.0
        denoise_loss_epoch = 0.0
        band_loss_epoch = 0.0
        rec_loss_epoch = 0.0
            
        device = self.accelerator.device
        for step, batch in enumerate(data_loader):
            # ================== 原 _train_step 的全部内容 ================== #
            lr_seq = batch["lr_seq"].to(device, dtype=torch.float32, non_blocking=True) # [B,K,3,H//4,W//4], k 个 lr frame 
            hr_seq = batch["hr_seq"].to(device, dtype=torch.float32, non_blocking=True) # [B,K,3,H,W] k 个 hr frame 
            lr_seq_neighbors = split_seq(lr_seq) #
            hr_seq_neighbors = split_seq(hr_seq)
            center_idx = self.cfg.K_neighbors // 2  # 固定中心帧索引
            
            hr_center = hr_seq_neighbors[center_idx].to(device, non_blocking=True)
            lr_seq_neighbors = [x.to(device, non_blocking=True) for x in lr_seq_neighbors] # 

            
            with torch.no_grad():
                x0_center = vae_encode_img(self.pipe, hr_center) # [B,4,hr_H//4,hr_W//4]
            B = lr_seq.shape[0]
            # t_int = sample_timesteps(self.pipe.scheduler, x0_center.shape[0], device)
            t_int = sample_timesteps(self.pipe.scheduler, 1, device)
            t_int = t_int.expand(B)
            center_z_t, center_target, a_bar = construct_noisy_latent_and_target(x0_center, t_int, self.pipe.scheduler, device)
            
            with self.accelerator.accumulate(self.pipe.unet):
                center_pred = self.training_validation_step(
                    t_int=t_int,
                    a_bar=a_bar,
                    lr_seq_neighbors=lr_seq_neighbors,
                    hr_seq_neighbors=hr_seq_neighbors,
                    center_idx=center_idx,
                    center_z_t=center_z_t,
                    is_train=is_train,
                )
                center_pred = center_pred.to(torch.float32)
                center_target = center_target.to(torch.float32)
                loss_denoise = F.mse_loss(center_pred, center_target)

                denoise_loss_epoch += loss_denoise.item()
                loss = loss_denoise

                if use_rec_loss or use_band_consistency_loss:
                    with self.autocast_context:
                        # ε→x0
                        if self.pipe.scheduler.config.prediction_type == "epsilon":
                            center_x0_hat = x0_from_eps(center_z_t.to(torch.float32),
                                            center_pred,
                                            a_bar.to(torch.float32)).to(torch.float32)
                        elif self.pipe.scheduler.config.prediction_type == "v_prediction":
                            center_x0_hat = x0_from_vpred(center_z_t.to(torch.float32),
                                            center_pred,
                                            a_bar.to(torch.float32)).to(torch.float32)
                        else:
                            raise ValueError(f"Unknown prediction_type: {self.pipe.scheduler.config.prediction_type}")
                        center_hr_pred = vae_decode_latent(self.pipe, center_x0_hat).to(torch.float32)  # 不记录显存
                        center_hr_pred = center_hr_pred.clamp(0, 1)

                        if self.cfg.use_band_consistency:
                            loss_band = self.band_reg(center_hr_pred, hr_center.to(torch.float32))
                            band_loss_epoch += loss_band.item()
                        else:
                            loss_band = torch.zeros(1, device=device)
                                
                        if use_rec_loss:
                            loss_rec = F.l1_loss(center_hr_pred, hr_center.to(torch.float32))
                        else:
                            loss_rec = torch.zeros(1, device=device)

                    band_loss_epoch += loss_band.item()
                    rec_loss_epoch += loss_rec.item()
                                        
                    loss = loss_denoise \
                            + self.cfg.band_weight * loss_band \
                            + self.cfg.rec_weight * loss_rec
                    
                else:
                    loss_band = torch.zeros(1, device=device)
                    loss_rec = torch.zeros(1, device=device)

                if is_train:
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                # ================== 训练逻辑结束 ================== #

            total_loss += loss.item()

            # 日志输出
            if self.accelerator.is_main_process and step % self.cfg.log_every == 0:
                self.logger.info(
                    f"[epoch {epoch} {mode} | step {step}] loss={loss.item():.4f} denoise loss={loss_denoise.item():.4f}, band loss = {loss_band.item():.4f}, rec loss = {loss_rec.item():.4f}",
                    main_process_only=self.accelerator.is_main_process
                )
                
 
                
            if step == 0 and epoch == self.start_epoch:
                ## print gpu memory usage
                max_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
                self.logger.info(f"[epoch {epoch} {mode}, Max GPU memory allocated after first step: {max_mem:.2f} GB]", main_process_only=self.accelerator.is_main_process)

            if self.debug and step >= 30:
                break
        
        del lr_seq, hr_seq, lr_seq_neighbors, hr_seq_neighbors, loss_denoise, loss_band, loss_rec, loss
        torch.cuda.empty_cache()
        
        return total_loss / max(1, len(data_loader)), denoise_loss_epoch / max(1, len(data_loader)), band_loss_epoch / max(1, len(data_loader)), rec_loss_epoch / max(1, len(data_loader))


    def train_loop(self):
        self.logger.info("Starting training loop...", main_process_only=self.accelerator.is_main_process)
        
        train_sum_loss_list = []
        train_denoise_loss_list = []
        train_band_loss_list = []
        train_rec_loss_list = []
        
        val_sum_loss_list = []
        val_denoise_loss_list = []
        val_band_loss_list = []
        val_rec_loss_list = []
        
        loss_curve_filepath = os.path.join(self.logging_dir, "loss_curve.jpg")
        for epoch in range(self.start_epoch, self.max_epochs):
            if epoch < 3:
                evaluate_every = 2
            elif epoch < 5:
                evaluate_every = 1
            else:
                evaluate_every = 1
            sum_loss, denoise_loss, band_loss, rec_loss = self.training_validation_epoch(self.train_loader, epoch, is_train=True, use_rec_loss=self.cfg.use_rec_loss, use_band_consistency_loss=self.cfg.use_band_consistency)
            train_sum_loss_list.append(sum_loss)
            train_denoise_loss_list.append(denoise_loss)
            train_band_loss_list.append(band_loss)
            train_rec_loss_list.append(rec_loss)
            if self.accelerator.is_main_process:
                self.logger.info(f"Train [Epoch {epoch}/{self.max_epochs}] avg_loss={sum_loss:.4f} denoise_loss={denoise_loss:.4f} band_loss={band_loss:.4f} rec_loss={rec_loss:.4f}", main_process_only=self.accelerator.is_main_process)

            self.save_checkpoint(epoch)
            if self.accelerator.is_main_process and self.val_loader is not None:
                sum_loss, denoise_loss, band_loss, rec_loss = self.training_validation_epoch(self.val_loader, epoch, is_train=False, use_rec_loss = True)
                val_sum_loss_list.append(sum_loss)
                val_denoise_loss_list.append(denoise_loss)
                val_band_loss_list.append(band_loss)
                val_rec_loss_list.append(rec_loss)
                self.logger.info(f"Val   [Epoch {epoch}/{self.max_epochs}] avg_loss={sum_loss:.4f} denoise_loss={denoise_loss:.4f} band_loss={band_loss:.4f} rec_loss={rec_loss:.4f}", main_process_only=self.accelerator.is_main_process)
                # print('rec_loss:', rec_loss)
                # print('best_loss:', self.best_loss)
                if rec_loss < self.best_loss:
                    self.best_loss = rec_loss
                    self.save_checkpoint(epoch, is_best=True, rec_loss=rec_loss)
                    self.logger.info(f"[CKPT] New best model saved with rec_loss={rec_loss:.4f}", main_process_only=self.accelerator.is_main_process)

            self.logger.info(f"\n", main_process_only=self.accelerator.is_main_process)
            
            ## plot loss curves
            if self.accelerator.is_main_process:
                plot_loss_curves(
                    train_sum_loss_list,
                    val_sum_loss_list,
                    train_denoise_loss_list,
                    val_denoise_loss_list,
                    train_band_loss_list,
                    val_band_loss_list,
                    train_rec_loss_list,
                    val_rec_loss_list,
                    loss_curve_filepath,
                    self.start_epoch
                )
            
            if self.debug and epoch >= 3:
                break


        self.logger.info("Training completed.", main_process_only=self.accelerator.is_main_process) 

    
    @torch.no_grad()
    def eval_step_per_frame(self, lr_neighbors: List[torch.Tensor], num_inference_steps=50, step = None):
        device = self.accelerator.device
        center_idx = len(lr_neighbors) // 2
        B, _, h, w = lr_neighbors[0].shape
        # print(f'lr_neighbors.shape: {[x.shape for x in lr_neighbors]}')

        num_neighbors = len(lr_neighbors)
        # latent 初始为高斯噪声
        z = torch.randn(B, 4, h, w, device=device, dtype=torch.float32)
        # print("init latent:", z.shape)   # torch.Size([2, 4, 32, 32])

        # 调度器设置要走的时间步
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps  # e.g. [999, 980, ..., 0]        
        alphas_cumprod = self.pipe.scheduler.alphas_cumprod.to(device)

        debug = True
        import tqdm
        for t in tqdm.tqdm(timesteps, desc="DDIM Inference", disable=not self.accelerator.is_main_process):
            # ---- 1) scheduler 用标量时间步；UNet 用 batch 版本 ----
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            t_batch  = torch.full((B,), t_scalar, device=device, dtype=torch.long)

            # ---- 2) ᾱ_t（按 batch 取索引）----
            a_bar_t = alphas_cumprod[t_batch].clamp(1e-6, 1. - 1e-6)

            # ---- 3) 规范化 latent 作为 UNet 输入（只对 z 做一次 scale）----
            z_scaled = self.pipe.scheduler.scale_model_input(z.to(self.pipe.unet.dtype), t_scalar)

            # ---- 4) 为每个邻帧 j 预测 ε̂_{j,t} or v̂_{j,t}（用同一个 z_scaled）----

                
            lr_i = lr_neighbors[center_idx]
            lr_i_norm = lr_i * 2 - 1

            lr_i_lat = F.interpolate(lr_i_norm, size=z.shape[-2:], mode="bilinear", align_corners=False)
            z_in_i = torch.cat([z_scaled, lr_i_lat.to(dtype=z_scaled.dtype)], dim=1)
            z_in_i = z_in_i.to(dtype=self.pipe.unet.dtype)

            # z_in_i = torch.cat([z, lr_i_lat], dim=1)
            pred_out_i = self.pipe.unet(
                sample=z_in_i,
                timestep=t_batch,
                encoder_hidden_states=self.null_text_emb[:B],
                class_labels=None
            ).sample

                

            # ---- 6) scheduler 一步（用原始 z，不是 z_scaled；时间步用 t_scalar）----
            z = self.pipe.scheduler.step(
                model_output=pred_out_i.to(torch.float32),
                timestep=t,
                sample=z
            ).prev_sample

        # print("latent after step:", z.shape) #torch.Size([2, 4, 32, 32])
        #  解码回图像
        sr_center = vae_decode_latent(self.pipe, z.to(self.pipe.unet.dtype))
        return sr_center, center_idx

    @torch.no_grad()
    def eval_step_per_sequence(self, lr_seq_path, hr_seq_path, num_inference_steps=75, step = 0):
        """
        lr_seq_full: # [B, T] 的 LR img filepath 列表
        hr_seq_path : # [B, T] 的 HR img filepath 列表
        return: List[T] of SR results, each [B,3,H,W]
        """

        T = len(lr_seq_path[0])
        PSNRs = []
        device = self.accelerator.device
        lr_scale = 4 if self.cfg.dataset_name == 'Vimeo-90K' else 1
        hr_seq_path_transposed = list(zip(*hr_seq_path))  # 转为 [T, B]
        for i in range(T):
            # 取得以第 i 帧为中心的 K 邻域
            lr_seq = pick_neighbors_for_eval(lr_seq_path, i, device, to_neg1_pos1=False, scale=lr_scale)
            # hr_neighbors = pick_neighbors_for_eval(hr_seq_path, i, device, dataset_name=self.cfg.dataset_name, to_neg1_pos1=False)
            sr_gt = img_list_to_tensor(hr_seq_path_transposed[i], to_neg1_pos1=False).to(device, dtype=torch.float32, non_blocking=True)
            
            lr_neighbors = split_seq(lr_seq)
            sr_pred, _ = self.eval_step_per_frame(lr_neighbors, num_inference_steps, step = step)
            sr_pred = sr_pred.clamp(0, 1).to(torch.float32)

            psnr = self._psnr(sr_pred, sr_gt)
            PSNRs.append(psnr)

            if self.debug and i >= 2:
                break
                                
            if self.dump_sr and self.accelerator.is_main_process:
                sr_img_np = (sr_pred[0].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
                img_pil = Image.fromarray(sr_img_np)
                img_pil.save(os.path.join(self.dump_dir, f"{step:03d}_{i:03d}_SR.png"))

                lr_img_np = (sr_gt[0].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
                img_pil = Image.fromarray(lr_img_np)
                img_pil.save(os.path.join(self.dump_dir, f"{step:03d}_{i:03d}_HR.png"))
                self.logger.info(f"[eval step {step} frame {i}] PSNR: {psnr:.2f} dB", main_process_only=self.accelerator.is_main_process)
            if self.debug and i >= 3:
                break
                
        return PSNRs


    @torch.no_grad()
    def evaluate(self, num_inference_steps=50):
        self.pipe.unet.eval()
        self.pipe.vae.eval()
        
        PSNRs = []
        self.logger.info(f"Starting evaluation with num_inference_steps={num_inference_steps}...", main_process_only=self.accelerator.is_main_process)
        device = self.accelerator.device
        for step, batch in enumerate(self.test_loader):
            lr_seq_path = batch["lr_seq_path"] # [T, B] 的lr img file path 列表
            hr_seq_path = batch["hr_seq_path"] # [B, T] 的hr img file path 列表
            lr_seq_path = list(zip(*lr_seq_path))  # 转为 [B, T]
            hr_seq_path = list(zip(*hr_seq_path))  # 转为 [B, T]
            # print('lr_seq_path', lr_seq_path)
            # print('hr_seq_path', hr_seq_path)
            psnr = self.eval_step_per_sequence(lr_seq_path, hr_seq_path, num_inference_steps, step)  # List[T]

            PSNRs += psnr
            # 日志输出
            if self.accelerator.is_main_process and step % self.cfg.log_every == 0:
                self.logger.info(f"[step {step}] Current PSNR: {psnr[-1]:.2f} dB",main_process_only=self.accelerator.is_main_process)

            if self.debug and step >= 5:
                break
                        
        avg_psnr = sum(PSNRs)/len(PSNRs)
        self.logger.info(f"[EVAL] PSNR over {len(PSNRs)} clips at num_inference_steps {num_inference_steps}: {avg_psnr:.2f} dB", main_process_only=self.accelerator.is_main_process)
        return avg_psnr

# =======================
#         ENTRY
# =======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with reduced settings')
    parser.add_argument('--runtime', type=str, default='train', help='Runtime mode: train or test')
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset to use for training/testing')
    parser.add_argument('--finetune', action='store_true', help='Whether to finetune from a pre-trained checkpoint')
    parser.add_argument('--unet_weights_precision', type=str, default=None, help='Precision for UNet weights: fp16 or bf16')
    parser.add_argument('--dump_sr', action='store_true', help='Dump super-resolved videos during evaluation')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')

    args = parser.parse_args()
    args.dump_sr = True  # 强制开启 dump_sr 功能以保存结果
    
    trainer = SD_X4_worker(args)
    if args.runtime == 'train':
        trainer.train_loop()
    else:
        trainer.evaluate()
