from diffusers import DiffusionPipeline
import torch
import config

base_sdxl_pipeline = DiffusionPipeline.from_pretrained(config.CHECKPOINT_PATH, 
                                         torch_dtype=torch.float16, 
                                         use_safetensors=True, 
                                         variant="fp16")
base_sdxl_pipeline.set_progress_bar_config(disable=True)
base_sdxl_pipeline.to("cuda")
