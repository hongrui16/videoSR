"""
Test-only script: directly use stabilityai/stable-diffusion-x4-upscaler
No training, no FRI / DSTCM / Wrapper.
Just load the pretrained model and upscale LR images.
"""

import os
import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import argparse
from tqdm import tqdm


def load_image(img_path, target_size=(128, 128)):
    """
    读取图像并缩小到低清尺寸（例如 128×128）
    输入输出范围保持在 [0,1]
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size, Image.BICUBIC)
    return img


def save_image(tensor, save_path):
    """
    将 [C,H,W] 或 [B,C,H,W] 的 tensor 存成图片
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    img = (tensor * 255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(img).save(save_path)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载预训练 Stable Diffusion ×4 超分模型
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.enable_attention_slicing()

    # 2. 创建输出文件夹
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. 处理输入图像路径或文件夹
    if os.path.isdir(args.input_path):
        img_list = [
            os.path.join(args.input_path, f)
            for f in os.listdir(args.input_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
    else:
        img_list = [args.input_path]

    # 4. 逐张图像超分
    for img_path in tqdm(img_list, desc="Upscaling images"):
        # (1) 加载低清图像，缩小至 128×128
        lr_img = load_image(img_path, target_size=(args.lr_size, args.lr_size))
        
        # (2) 推理，text_prompt 可以为空字符串或自定义
        with torch.no_grad():
            result = pipe(
                prompt=args.prompt,
                image=lr_img,
            ).images[0]   # PIL Image

        # (3) 保存输出结果
        if args.prompt.strip() == "":
            save_name = os.path.splitext(os.path.basename(img_path))[0] + "_sr_no_prompt.png"
        else:
            save_name = os.path.splitext(os.path.basename(img_path))[0] + "_sr_prompt.png"
        save_path = os.path.join(args.output_dir, save_name)
        result.save(save_path)
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="lr_images",
                        help="Path to a single LR image or a folder of images")
    parser.add_argument("--output_dir", type=str, default="output_SR",
                        help="Directory to save SR results")
    parser.add_argument("--lr_size", type=int, default=128,
                        help="Resize input image to this size (e.g. 128x128) before upscaling")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt to guide the upscaling (can be empty)")
    args = parser.parse_args()

    main(args)
