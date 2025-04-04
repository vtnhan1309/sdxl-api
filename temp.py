from diffusers import DiffusionPipeline
import torch

base_sdxl_pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                         torch_dtype=torch.float16, 
                                         use_safetensors=True, 
                                         variant="fp16")
output_dir = '/network-volume/sd/sdxl-base'
base_sdxl_pipeline.save_pretrained(output_dir, safe_serialization=True)
