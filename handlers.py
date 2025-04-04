import torch
from diffusers import DiffusionPipeline
from models import ImageRequest, ImageResponse, ImageChoicesData
from utils import image_to_base64


def handler_image_generation(pipeline: DiffusionPipeline, req: ImageRequest) -> ImageResponse:
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
    data = []
    for idx, image in enumerate(images):
        data.append(ImageChoicesData(
            index=idx + 1,
            b64_json=image_to_base64(image),
            url=''
        ))
    response = ImageResponse(
        id='',
        model=req.model,
        data=data
    )
    torch.cuda.empty_cache()
    return response
