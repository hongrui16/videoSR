from peft import LoraConfig, inject_adapter_in_model



def enable_lora(unet, r=8, lora_alpha=16, logger=None, target_modules=None):
    """
    为 SD-x4 超分任务优化的 LoRA 配置
    
    Args:
        r: LoRA rank，推荐 8-16
        lora_alpha: 缩放因子，通常 = r 或 2*r
        target_modules: 要应用 LoRA 的模块
    """
    if target_modules is None:
        # 只对 cross-attention 应用（最高效）
        target_modules = [
            "attn2.to_q",  # cross-attn Q
            "attn2.to_k",  # cross-attn K  
            "attn2.to_v",  # cross-attn V
        ]
    
    
    # 方法1：用 PEFT（推荐，最简单）
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        init_lora_weights=True,
    )
    
    # 注入 LoRA
    unet = inject_adapter_in_model(lora_config, unet)
    
    # 统计
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    
    if logger is None:
        print(
            f"[LoRA] Rank={r}, Alpha={lora_alpha}\n"
            f"       Trainable: {trainable:,} / {total:,} "
            f"({100*trainable/total:.2f}%)\n"
            f"       Memory saved: ~{(total-trainable)*4/1e9:.1f} GB"
        )
    else:
        logger.info(
            f"[LoRA] Rank={r}, Alpha={lora_alpha}\n"
            f"       Trainable: {trainable:,} / {total:,} "
            f"({100*trainable/total:.2f}%)\n"
            f"       Memory saved: ~{(total-trainable)*4/1e9:.1f} GB"
        )
    
    return unet



def verify_lora_injected(unet):
    """验证 LoRA 是否真的注入到 UNet"""
    lora_layers = []

    for name, module in unet.named_modules():
        # 检查是否包含 lora 相关层
        if any(keyword in name.lower() for keyword in ["lora", "adapter"]):
            lora_layers.append(name)
    
    if lora_layers:
        print(f"✅ Found {len(lora_layers)} LoRA layers")
        print(f"   Examples: {lora_layers[:3]}")
        return True
    else:
        print("❌ No LoRA layers found!")
        return False



def verify_lora_gradients(unet):
    """验证只有 LoRA 参数会更新"""
    trainable_names = []
    frozen_names = []

    for name, param in unet.named_parameters():
        if param.requires_grad:
            trainable_names.append(name)
        else:
            frozen_names.append(name)
    
    print(f"✅ Trainable params: {len(trainable_names)}")
    print(f"   Examples: {trainable_names[:3]}")
    
    print(f"✅ Frozen params: {len(frozen_names)}")
    
    # 确保所有可训练的都是 LoRA
    non_lora_trainable = [n for n in trainable_names if "lora" not in n.lower()]
    if non_lora_trainable:
        print(f"⚠️  Warning: Non-LoRA trainable params found: {non_lora_trainable[:3]}")
    else:
        print("✅ All trainable params are LoRA!")
    
    return len(non_lora_trainable) == 0




def log_module_info(wrapper, logger):
    """
    打印关键模块的 device / dtype / 是否参与训练（requires_grad）
    """
    
    # 如果wrapper被DDP包裹，取出原始模型
    wrapper = wrapper.module if hasattr(wrapper, "module") else wrapper
    pipe = wrapper.pipe
    modules = {
    "UNet": pipe.unet,
    "VAE": pipe.vae,
    "Conditioner": getattr(wrapper, "conditioner", None),
    "Injector": getattr(wrapper, "injector", None),
    "FlowEstimator": getattr(wrapper, "flow_estimator", None),
    "QRM": getattr(getattr(wrapper, "ref_builder", None), "qrm", None),
        }


    logger.info("\n===== Module Summary =====")
    for name, module in modules.items():
        if module is None:
            continue

        # 获取第一个参数判断 device / dtype / grad
        try:
            p = next(module.parameters())
            device = p.device
            dtype = p.dtype
            requires_grad = any(param.requires_grad for param in module.parameters())
        except StopIteration:
            device, dtype, requires_grad = None, None, None

        logger.info(
            f"{name:<15} | device={device} | dtype={dtype} | trainable={requires_grad}"
        )
