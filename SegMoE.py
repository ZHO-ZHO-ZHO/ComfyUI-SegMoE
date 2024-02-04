import os
import cv2
import torch
import numpy as np
from PIL import Image
import folder_paths

from huggingface_hub import hf_hub_download
from .segmoe import SegMoEPipeline

current_directory = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "cpu"


class SMoE_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config_or_path": ("STRING", {"default": "segmind/SegMoE-4x2-v0"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "🎩SegMoE"
  
    def load_model(self, config_or_path):
        # Code to load the base model
        pipe = SegMoEPipeline(
            config_or_path,
            device = device,
        )
        return [pipe]


class SMoE_Generation_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("MODEL",),
                "positive": ("STRING", {"default": "cosmic canvas, orange city background, painting of a chubby cat", "multiline": True}),
                "negative": ("STRING", {"default": "nsfw, bad quality, worse quality", "multiline": True}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1, "display": "slider"}),
                "guidance_scale": ("FLOAT", {"default": 5, "min": 0, "max": 10, "display": "slider"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32, "display": "slider"}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 32, "display": "slider"}), 
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "smoe_generate_image"
    CATEGORY = "🎩SegMoE"
                       
    def smoe_generate_image(self, pipe, positive, negative, steps, guidance_scale, seed, width, height):
            
        generator = torch.Generator(device=device).manual_seed(seed)

        output = pipe(
            prompt=positive,
            negative_prompt=negative,
            num_inference_steps=steps,
            generator=generator,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
            return_dict=False
            )

        # 检查输出类型并相应处理
        if isinstance(output, tuple):
            # 当返回的是元组时，第一个元素是图像列表
            images_list = output[0]
        else:
            # 如果返回的是，需要从中提取图像
            images_list = output.images

        # 转换图像为 torch.Tensor，并调整维度顺序为 NHWC
        images_tensors = []
        for img in images_list:
            # 将 PIL.Image 转换为 numpy.ndarray
            img_array = np.array(img)
            # 转换 numpy.ndarray 为 torch.Tensor
            img_tensor = torch.from_numpy(img_array).float() / 255.
            # 转换图像格式为 CHW (如果需要)
            if img_tensor.ndim == 3 and img_tensor.shape[-1] == 3:
                img_tensor = img_tensor.permute(2, 0, 1)
            # 添加批次维度并转换为 NHWC
            img_tensor = img_tensor.unsqueeze(0).permute(0, 2, 3, 1)
            images_tensors.append(img_tensor)

        if len(images_tensors) > 1:
            output_image = torch.cat(images_tensors, dim=0)
        else:
            output_image = images_tensors[0]

        return (output_image,)

NODE_CLASS_MAPPINGS = {
    "SMoE_ModelLoader_Zho": SMoE_ModelLoader_Zho,
    "SMoE_Generation_Zho": SMoE_Generation_Zho,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SMoE_ModelLoader_Zho": "🎩SegMoE Model Loader",
    "SMoE_Generation_Zho": "🎩SegMoE Generation",
}
