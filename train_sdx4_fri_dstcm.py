# train_sdx4_fri_dstcm_full.py (refactored)
# SD×4 + FRI + DS-TCM —— diffusion ε-target
# - UNet/VAE frozen; train conditioner (+optional QRM)
# - Supports dataloader: list[T * (B,3,H,W)] or tensor (B,T,3,H,W)
# - Single cohesive Trainer class; no duplicated functions

import os
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

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


# from config.config import TrainConfig
from dataloader.build_dataloader import build_dataloader
from cond_modules import SDX4_FRI_DSTCM_Wrapper
from utils.loss import BandConsistencyLoss
from config.config import TrainConfig

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

def prepare_pipe(cfg, device):
    if cfg.unet_weights_precision == "fp32":
        weight_dtype = torch.float32
    elif cfg.unet_weights_precision == "bf16":
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


    # freeze backbone
    pipe.unet.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    if hasattr(pipe, "text_encoder"):
        pipe.text_encoder.requires_grad_(False)
        pipe.text_encoder.to(device)     
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(device=device, dtype=weight_dtype)
    pipe.unet.eval(); pipe.vae.eval()
    return pipe

@torch.no_grad()
def vae_encode_img(pipe, img_01: torch.Tensor) -> torch.Tensor:
    """ img_01 in [0,1], shape [B,3,H,W] -> latent [B,4,h,w] """
    # print("vae_encode_img input img_01:", img_01.shape) # [B,3,H,W]
    img = img_01 * 2 - 1
    posterior = pipe.vae.encode(img).latent_dist
    z = posterior.mean
    # print("vae_encode_img output z:", z.shape) # [B,4,H//4,W//4]
    return z * pipe.vae.config.scaling_factor

@torch.no_grad()
def vae_decode_latent(pipe, z_4chw: torch.Tensor) -> torch.Tensor:
    """ latent [B,4,h,w] -> img_01 in [0,1], shape [B,3,H,W] """
    z_4chw = z_4chw.to(dtype=pipe.vae.dtype)
    out = pipe.vae.decode(z_4chw / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return (out.clamp(-1, 1) + 1) / 2.0


# =======================
#   SEQUENCE UTILITIES
# =======================

def split_seq(x: Union[List[torch.Tensor], torch.Tensor]) -> List[torch.Tensor]:
    if isinstance(x, list):
        return x
    # x: [B,T,3,H,W]
    return [x[:, t] for t in range(x.shape[1])]

def pick_neighbors(seq: List[torch.Tensor], K: int, center_mode="middle") -> Tuple[List[torch.Tensor], int]:
    T = len(seq)
    if center_mode == "fixed0":
        c = 0
    elif center_mode == "random":
        c = torch.randint(0, T, (1,)).item()
    else:
        c = T // 2
    idxs = [c]
    l = r = 1
    while len(idxs) < K and (c - l >= 0 or c + r < T):
        if c - l >= 0:
            idxs.append(c - l); l += 1
        if len(idxs) < K and c + r < T:
            idxs.append(c + r); r += 1
    idxs = sorted(idxs)
    return [seq[i] for i in idxs], idxs.index(c)


# =======================
#   DIFFUSION OBJECTIVE
# =======================

def sample_timesteps(scheduler, bsz: int, device: torch.device) -> torch.Tensor:
    num_steps = scheduler.config.num_train_timesteps
    return torch.randint(low=0, high=num_steps, size=(bsz,), device=device, dtype=torch.long)





def construct_noisy_latent_and_etarget(x0: torch.Tensor, t_int: torch.Tensor, scheduler, device):
    """
    returns: z_t, eps_target, a_bar
    """
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    a_bar = alphas_cumprod[t_int].clamp(1e-6, 1.0 - 1e-6)  # [B]
    B = x0.shape[0]
    eps = torch.randn_like(x0)
    z_t = a_bar.sqrt().view(B,1,1,1) * x0 + (1. - a_bar).sqrt().view(B,1,1,1) * eps
    eps_target = eps
    return z_t, eps_target, a_bar

# =======================
#        TRAINER
# =======================

class FRI_DSTCM_Trainer:
    def __init__(self, args: argparse.Namespace):
        
        cfg = TrainConfig()
        self.cfg = cfg        
        self.args = args
        
        self.debug = args.debug        
        self.runtime_mode = cfg.runtime_mode  # "train" | "test"
        self.max_epochs = cfg.max_epochs
        self.finetune = cfg.finetune

        if cfg.data_precision == "fp16":
            self.data_precision = torch.float16
        elif cfg.data_precision == "bf16":
            self.data_precision = torch.bfloat16
        else:
            self.data_precision = torch.float32

        current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        job_id = str(os.getpid())
        network_name = cfg.network_type
        dataset_name = args.dataset_name if args.dataset_name is not None else cfg.dataset_name
        
        
        if self.debug:
            self.logging_dir = os.path.join(cfg.log_dir, network_name, dataset_name, 'debug_' + current_timestamp + '_' + job_id)
        else:
            self.logging_dir = os.path.join(cfg.log_dir, network_name, dataset_name, current_timestamp + '_' + job_id)
        # self.output_vis_dir = os.path.join(self.logging_dir, "vis")                    
        if self.debug:
            self.ckpt_out_dir = self.logging_dir
        else:
            self.ckpt_out_dir = os.path.join(cfg.ckpt_out_dir, network_name, dataset_name, current_timestamp + '_' + job_id)

        
        accelerator_project_config = ProjectConfiguration(project_dir=self.logging_dir, logging_dir=self.logging_dir)
        
        self.accelerator = Accelerator(mixed_precision=self.cfg.mixed_precision, 
                                       gradient_accumulation_steps=self.cfg.grad_accum,
                                        project_config=accelerator_project_config,
                                        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]                    
                                   )
        self.accelerator.wait_for_everyone()
        
        self.autocast_context = get_autocast_context(self.accelerator)

        if self.accelerator.is_main_process:
            os.makedirs(self.logging_dir, exist_ok=True)
            if self.runtime_mode == 'train':
                os.makedirs(self.ckpt_out_dir, exist_ok=True)
                
        log_dir_last_name = os.path.basename(os.path.normpath(self.logging_dir))

        log_path = os.path.join(self.logging_dir, f"{log_dir_last_name}_info.log")
        
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
        
        
        device = self.accelerator.device
                        
        self.pipe = prepare_pipe(cfg, device)
        self.tokenizer = self.pipe.tokenizer
        self.text_encoder = getattr(self.pipe, "text_encoder", None)
        self.text_encoder.to(device) if self.text_encoder is not None else None
        
        B = self.cfg.train_bsz    
        if self.text_encoder is not None:
            with torch.no_grad():
                tokens = self.tokenizer(
                    [""] * B,  # 空文本
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

        self.wrapper = SDX4_FRI_DSTCM_Wrapper(self.pipe, use_qrm=self.cfg.use_qrm, K=self.cfg.K_neighbors,
                                              device =device,
                                              null_text_emb=self.null_text_emb,
                                                logger=self.logger,
                                                accelerator=self.accelerator
                                              ).to(device)
        self.wrapper.to(device)  
        self.logger.info(f"Wrapper created. use_qrm={self.cfg.use_qrm}, K_neighbors={self.cfg.K_neighbors}", main_process_only=self.accelerator.is_main_process)
        self.band_reg = BandConsistencyLoss().to(device) if self.cfg.use_band_consistency else None

        params = list(self.wrapper.conditioner.parameters())
        if self.cfg.use_qrm and getattr(self.wrapper.ref_builder, "qrm", None) is not None:
            params += list(self.wrapper.ref_builder.qrm.parameters())
        self.optimizer = torch.optim.AdamW(params, lr=self.cfg.lr)

        if self.debug:
            self.cfg.train_bsz = 2
                
        self.train_loader, self.train_dataset = build_dataloader(
            split="train",
            batch_size=self.cfg.train_bsz,
            num_workers=self.cfg.num_workers,
            dataset_name=dataset_name,
            device=device,
        )
        
        
        self.logger.info(f"Train dataset size: {len(self.train_dataset)}", main_process_only=self.accelerator.is_main_process)
        self.test_loader, self.test_dataset = build_dataloader(
            split="test",
            batch_size=self.cfg.train_bsz,
            num_workers=self.cfg.num_workers,
            dataset_name=dataset_name,
            device=device,
        )
        self.logger.info(f"Test dataset size: {len(self.test_dataset)}", main_process_only=self.accelerator.is_main_process)
        
        prepared = [self.wrapper, self.optimizer, self.train_loader]
        self.wrapper, self.optimizer, self.train_loader = self.accelerator.prepare(*prepared)
        if self.test_loader is not None:
            (self.test_loader) = self.accelerator.prepare(self.test_loader)

        self.logger.info(f"Data loaders prepared.", main_process_only=self.accelerator.is_main_process)

        # optional resume
        self.start_epoch = 0
        self.best_metric = float("-inf")
        if self.cfg.resume_from:
            with self.accelerator.main_process_first():
                self.start_epoch = self.load_checkpoint(self.cfg.resume_from, map_location=self.accelerator.device) + 1
            self.accelerator.wait_for_everyone()
            self.logger.info(f"Resumed from {self.cfg.resume_from} at epoch {self.start_epoch}", main_process_only=self.accelerator.is_main_process)

        

    # ---------- checkpoint ----------
    def save_checkpoint(self, epoch, is_best = False, psnr : Optional[float] = None):
        if not self.accelerator.is_main_process:
            return
        os.makedirs(self.ckpt_out_dir, exist_ok=True)
        # tag = tag or f"step_{self.step}"
        if is_best:
            path = os.path.join(self.ckpt_out_dir, f"fri_dstcm_best.pt")
        else:
            path = os.path.join(self.ckpt_out_dir, f"fri_dstcm.pt")

        # unwrap 以兼容 DDP/FSDP/DeepSpeed
        real_wrapper = self.accelerator.unwrap_model(self.wrapper)

        ckpt = {
            "conditioner": real_wrapper.conditioner.state_dict(),
            "qrm": (
                getattr(real_wrapper.ref_builder, "qrm", None).state_dict()
                if getattr(real_wrapper.ref_builder, "qrm", None) is not None
                else None
            ),
            "optimizer": self.optimizer.state_dict(),  # accelerate 处理过也能保存
            "epoch": epoch,
            "psnr": psnr,
        }

        
        torch.save(ckpt, path)
        self.logger.info(f"[CKPT] saved -> {path}")


    def load_checkpoint(self, ckpt_path: str, map_location: str = "cpu") -> int:
        if not os.path.isfile(ckpt_path):
            self.logger.warning(f"[CKPT] No checkpoint found at {ckpt_path}")
            return -1
        
        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.logger.info(f"[CKPT] loaded <- {ckpt_path}")

        # unwrap 以兼容分布式封装
        real_wrapper = self.accelerator.unwrap_model(self.wrapper)

        # 载入 conditioner
        missing, unexpected = real_wrapper.conditioner.load_state_dict(
            ckpt["conditioner"], strict=False
        )
        if missing or unexpected:
            self.logger.warning(f"[CKPT] conditioner missing={missing}, unexpected={unexpected}", main_process_only=self.accelerator.is_main_process)

        # 载入 QRM（若存在）
        if ckpt.get("qrm") is not None and getattr(real_wrapper.ref_builder, "qrm", None) is not None:
            _m, _u = real_wrapper.ref_builder.qrm.load_state_dict(ckpt["qrm"], strict=False)
            if _m or _u:
                self.logger.warning(f"[CKPT] qrm missing={_m}, unexpected={_u}", main_process_only=self.accelerator.is_main_process)

        if self.finetune:
            return -1
        
        # 载入优化器（若在训练中恢复）
        if self.optimizer is not None and ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.logger.info(f"[CKPT] optimizer loaded <- {ckpt_path}")

        if self.best_metric is not None:
            self.best_metric = ckpt.get("psnr", float("-inf"))
            self.logger.info(f"[CKPT] best_metric loaded <- {ckpt_path}")
            
        return int(ckpt.get("epoch", 0))

    def _unet_step_with_injection(
        self,
        t_int: torch.Tensor,
        a_bar: torch.Tensor,
        lr_seq_neighbors: List[torch.Tensor],
        hr_seq_neighbors: List[torch.Tensor],
        center_idx: int,
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
        z_t_list, pred_eps_list = [], []

        # ===== (1) Teacher-forcing 所有邻帧 j =====
        with torch.no_grad(), self.autocast_context:
            for lr_j, hr_j in zip(lr_seq_neighbors, hr_seq_neighbors):
                # encode & 加噪
                x0_j = vae_encode_img(self.pipe, hr_j)
                eps_j = torch.randn_like(x0_j)
                z_j_t = a_bar.sqrt().view(B,1,1,1)*x0_j + (1.-a_bar).sqrt().view(B,1,1,1)*eps_j
                # 拼接 LR → 7ch
                lr_lat = F.interpolate(lr_j, size=z_j_t.shape[-2:], mode="bilinear", align_corners=False)
                z_in = torch.cat([z_j_t, lr_lat], dim=1)
                # 预测 ε̂_{j,t}
                pred_eps_j = self.pipe.unet(sample=z_in, timestep=t_int, encoder_hidden_states=self.null_text_emb[:z_in.shape[0]], class_labels=None).sample
                z_t_list.append(z_j_t)
                pred_eps_list.append(pred_eps_j)

        try:

            # ===== (2) warp 邻帧 → cond_i 并注册注入 =====
            self.wrapper.build_and_register(
                z_t_list=z_t_list,
                lr_rgb_seq=lr_seq_neighbors,
                pred_eps_list=pred_eps_list,
                alpha_bar_t=a_bar,
                timestep=t_int,
                center_idx=center_idx,
            )

            # ===== (3) 计算中心帧 i 的 ε̂_{i,t} =====
            with self.autocast_context:
                z_i_t = z_t_list[center_idx]
                lr_i = lr_seq_neighbors[center_idx]
                lr_i_lat = F.interpolate(lr_i, size=z_i_t.shape[-2:], mode="bilinear", align_corners=False)
                z_i_7ch = torch.cat([z_i_t, lr_i_lat], dim=1)
                pred_eps_i_t = self.pipe.unet(sample=z_i_7ch, timestep=t_int, encoder_hidden_states=self.null_text_emb[:z_i_7ch.shape[0]], class_labels=None).sample
        finally:
            # ===== (4) 清理 hook =====
            self.wrapper.clear()
        return pred_eps_i_t


    # ---------- eval ----------
    @torch.no_grad()
    def _psnr(self, x: torch.Tensor, y: torch.Tensor) -> float:
        mse = F.mse_loss(x, y, reduction="mean").item()
        if mse == 0: return 100.0
        return 10.0 * torch.log10(torch.tensor(1.0) / torch.tensor(mse)).item()

    @torch.no_grad()
    def _run_upscale_with_conditioning(self, lr_seq_full: List[torch.Tensor], num_inference_steps=50):
        device = lr_seq_full[0].device
        lr_neighbors, center_idx = pick_neighbors(lr_seq_full, self.wrapper.K, "middle")
        B, _, h, w = lr_neighbors[0].shape


        # latent 初始为高斯噪声
        z = torch.randn(B, 4, h, w, device=device, dtype=self.pipe.unet.dtype)
        # print("init latent:", z.shape)   # torch.Size([2, 4, 32, 32])

        # 调度器设置要走的时间步
        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps  # e.g. [999, 980, ..., 0]
        
        alphas_cumprod = self.pipe.scheduler.alphas_cumprod.to(device)

        for t in timesteps:
            # ---- 1) scheduler 用标量时间步；UNet 用 batch 版本 ----
            t_scalar = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            t_batch  = torch.full((B,), t_scalar, device=device, dtype=torch.long)

            # ---- 2) ᾱ_t（按 batch 取索引）----
            a_bar_t = alphas_cumprod[t_batch].clamp(1e-6, 1. - 1e-6)

            # ---- 3) 规范化 latent 作为 UNet 输入（只对 z 做一次 scale）----
            z_scaled = self.pipe.scheduler.scale_model_input(z, t_scalar)

            # ---- 4) 为每个邻帧 j 预测 ε̂_{j,t}（用同一个 z_scaled）----
            pred_eps_list = []
            for lr_j in lr_neighbors:
                lr_j_lat = F.interpolate(lr_j, size=z_scaled.shape[-2:], mode="bilinear", align_corners=False)
                z_in_j = torch.cat([z_scaled, lr_j_lat], dim=1)  # 7ch = [z(4) + LR↑(3)]
                pred_eps_j = self.pipe.unet(
                    sample=z_in_j,
                    timestep=t_batch,
                    encoder_hidden_states=self.null_text_emb[:B],
                    class_labels=None
                ).sample
                pred_eps_list.append(pred_eps_j)

            # ---- 5) 注册残差注入 → 预测中心帧的 ε（依然用自己的 LR_i↑）----
            try:
                self.wrapper.build_and_register(
                    z_t_list=[z] * self.wrapper.K,      # 注意：这里传未 scale 的 z 作为 z_t
                    lr_rgb_seq=lr_neighbors,
                    pred_eps_list=pred_eps_list,
                    alpha_bar_t=a_bar_t,
                    timestep=t_batch,
                    center_idx=center_idx,
                )

                lr_i = lr_neighbors[center_idx]
                lr_i_lat = F.interpolate(lr_i, size=z_scaled.shape[-2:], mode="bilinear", align_corners=False)
                z_in_i = torch.cat([z_scaled, lr_i_lat], dim=1)
                pred_eps_inj = self.pipe.unet(
                    sample=z_in_i,
                    timestep=t_batch,
                    encoder_hidden_states=self.null_text_emb[:B],
                    class_labels=None
                ).sample
            finally:
                self.wrapper.clear()

        # ---- 6) scheduler 一步（用原始 z，不是 z_scaled；时间步用 t_scalar）----
        z = self.pipe.scheduler.step(
            model_output=pred_eps_inj,
            timestep=t,
            sample=z
        ).prev_sample

        # print("latent after step:", z.shape) #torch.Size([2, 4, 32, 32])
        #  解码回图像
        I_pred = vae_decode_latent(self.pipe, z)
        return I_pred, center_idx

    def training_epoch(self, epoch):
        self.pipe.unet.train()
        total_loss = 0.0
        denoise_loss_epoch = 0.0
        band_loss_epoch = 0.0
        device = self.accelerator.device
        for step, batch in enumerate(self.train_loader):

            # ================== 原 _train_step 的全部内容 ================== #
            lr_seq = batch["lr_seq"].to(device, dtype=self.data_precision, non_blocking=True) # [B,T,3,H,W]
            hr_seq = batch["hr_seq"].to(device, dtype=self.data_precision, non_blocking=True) # [B,T,3,H,W]
            lr_seq_full = split_seq(lr_seq)
            hr_seq_full = split_seq(hr_seq)

            lr_seq_neighbors, center_idx = pick_neighbors(lr_seq_full, self.cfg.K_neighbors, self.cfg.center_idx_mode)
            hr_seq_neighbors, _          = pick_neighbors(hr_seq_full, self.cfg.K_neighbors, self.cfg.center_idx_mode)

            
            hr_center = hr_seq_full[center_idx].to(device, non_blocking=True)
            lr_seq_neighbors = [x.to(device, non_blocking=True) for x in lr_seq_neighbors]

            with torch.no_grad():
                x0 = vae_encode_img(self.pipe, hr_center)

            t_int = sample_timesteps(self.pipe.scheduler, x0.shape[0], device)
            z_t, eps_target, a_bar = construct_noisy_latent_and_etarget(x0, t_int, self.pipe.scheduler, device)

            with self.accelerator.accumulate(self.wrapper):
                pred_eps = self._unet_step_with_injection(
                    t_int=t_int,
                    a_bar=a_bar,
                    lr_seq_neighbors=lr_seq_neighbors,
                    hr_seq_neighbors=hr_seq_neighbors,
                    center_idx=center_idx,
                )

                loss_denoise = F.mse_loss(pred_eps, eps_target)
                denoise_loss_epoch += loss_denoise.item()
                loss = loss_denoise

                if self.cfg.use_band_consistency:
                    with self.autocast_context:
                        x0_hat = self.wrapper.x0_from_eps(z_t, pred_eps, a_bar)
                        x0_hat = x0_hat.to(dtype=self.data_precision)
                        hr_pred = vae_decode_latent(self.pipe, x0_hat)
                        loss_band = self.band_reg(hr_pred, hr_center)
                        band_loss_epoch += loss_band.item()
                    loss = loss + self.cfg.band_weight * loss_band

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            # ================== 训练逻辑结束 ================== #

            total_loss += loss.item()


            # 日志输出
            if self.accelerator.is_main_process and step % self.cfg.log_every == 0:
                self.logger.info(
                    f"[epoch {epoch} | step {step}] loss={loss.item():.4f} denoise loss={loss_denoise.item():.4f}, band loss = {loss_band.item() if self.cfg.use_band_consistency else 0:.4f}",
                    main_process_only=self.accelerator.is_main_process
                )            
                
            if step == 0 and epoch == self.start_epoch:
                ## print gpu memory usage
                max_mem = torch.cuda.max_memory_allocated(device=device) / (1024 ** 3)
                self.logger.info(f"Max GPU memory allocated after first step: {max_mem:.2f} GB", main_process_only=self.accelerator.is_main_process)

            if self.debug and step >= 20:
                break

        return total_loss / max(1, len(self.train_loader)), denoise_loss_epoch / max(1, len(self.train_loader)), band_loss_epoch / max(1, len(self.train_loader))


    def train_loop(self):
        self.logger.info("Starting training loop...", main_process_only=self.accelerator.is_main_process)
        
        for epoch in range(self.start_epoch, self.max_epochs):
            if epoch < 15:
                evaluate_every = 4
            elif epoch < 30:
                evaluate_every = 2
            else:
                evaluate_every = 1
            sum_loss, denoise_loss, band_loss = self.training_epoch(epoch)
            if self.accelerator.is_main_process:
                self.logger.info(f"[Epoch {epoch}/{self.max_epochs}] avg_loss={sum_loss:.4f} denoise_loss={denoise_loss:.4f} band_loss={band_loss:.4f}", main_process_only=self.accelerator.is_main_process)

            if self.accelerator.is_main_process and self.test_loader is not None and (epoch + 1) % evaluate_every == 0:
                
                avg_psnr = self.evaluate(self.test_loader)
                self.logger.info(f"[EVAL] PSNR: {avg_psnr:.2f} dB")
                if avg_psnr > self.best_metric:
                    self.best_metric = avg_psnr
                    self.save_checkpoint(epoch, is_best=True, psnr=avg_psnr)
                    self.logger.info(f"[CKPT] New best model saved with PSNR={avg_psnr:.2f} dB", main_process_only=self.accelerator.is_main_process)
            self.save_checkpoint(epoch)

            if self.debug :
                break

        self.logger.info("Training completed.", main_process_only=self.accelerator.is_main_process) 

    @torch.no_grad()
    def evaluate(self, dl_eval, num_inference_steps=50):
        self.pipe.unet.eval()
        PSNRs = []

        device = self.accelerator.device
        for step, batch in enumerate(dl_eval):
            lr_seq = batch["lr_seq"].to(device, dtype=self.data_precision, non_blocking=True)
            hr_seq = batch["hr_seq"].to(device, dtype=self.data_precision, non_blocking=True)
            lr_seq_full = split_seq(lr_seq)
            hr_seq_full = split_seq(hr_seq)
            # print('shape of lr_seq: ', lr_seq.shape) 
            # print('shape of hr_seq: ', hr_seq.shape)
            I_pred, c = self._run_upscale_with_conditioning(lr_seq_full, num_inference_steps)
            I_gt = hr_seq_full[c]
            # self.logger.info(f"I_pred size: {I_pred.shape}, I_gt size: {I_gt.shape}")

            PSNRs.append(self._psnr(I_pred, I_gt))

            if self.debug and step >= 10:
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
    parser.add_argument('--runtime_mode', type=str, default='train', choices=['train', 'test'], help='Runtime mode: train or test')
    parser.add_argument('--dataset_name', type=str, default=None, help='Name of the dataset to use for training/testing')
    
    args = parser.parse_args()
    trainer = FRI_DSTCM_Trainer(args)
    if args.runtime_mode == 'train':
        trainer.train_loop()
    else:
        trainer.evaluate()
