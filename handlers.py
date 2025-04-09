from typing import List
from PIL import Image
import torch
from diffusers import DiffusionPipeline, FluxPipeline
from models import ImageRequest, ImageResponse, ImageChoicesData
from utils import image_to_base64


def _build_image_generation_response(images: List[Image.Image], model:str) -> ImageResponse:
    data = []
    for idx, image in enumerate(images):
        data.append(ImageChoicesData(
            index=idx + 1,
            b64_json=image_to_base64(image),
            url=''
        ))
    response = ImageResponse(
        id='',
        model=model,
        data=data
    )
    return response


def handler_image_generation_sdxl(pipeline: DiffusionPipeline, req: ImageRequest) -> ImageResponse:
    generator = None
    if req.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(req.seed)
    try:
        images = pipeline(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_images_per_prompt=req.n,
            num_inference_steps=req.steps,
            generator=generator,
        ).images
    except Exception as e:
        print(f'Error at generating image: {e}')
        raise Exception('Internal server error')

    if len(images) == 0:
        raise Exception('Internal server error: empty images')
    response = _build_image_generation_response(images, req.model)
    torch.cuda.empty_cache()
    return response

def handler_image_generation_flux(pipeline: FluxPipeline, req: ImageRequest) -> ImageResponse:
    generator = None
    if req.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(req.seed)
    try:
        images = pipeline(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            height=req.height,
            width=req.width,
            num_images_per_prompt=req.n,
            num_inference_steps=req.steps,
            generator=generator,
            max_sequence_length=512,
            guidance_scale=3.5
        ).images
    except Exception as e:
        print(f'Error at generating image: {e}')
        raise Exception('Internal server error')

    if len(images) == 0:
        raise Exception('Internal server error: empty images')
    response = _build_image_generation_response(images, req.model)
    torch.cuda.empty_cache()
    return response
