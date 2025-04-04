import gc
import os
from fastapi import FastAPI
import bentoml
from diffusers import DiffusionPipeline
import torch
from bentoml.exceptions import UnprocessableEntity, InternalServerError
from typing_extensions import Union
from handlers import handler_image_generation
from models import ImageRequest
import config


# Create the FastAPI app instance
app = FastAPI()

# Define a route for the root URL
@app.get("/health")
def health_check():
    return {"message": "OK"}

@bentoml.service(
    traffic={"timeout": 300},
    workers=int(os.getenv('NUM_WORKERS', '1')),
    labels={'owner': 'greennode-team', 'project': 'sdxl'},
    resources={
        "gpu": 1,
    },
)
@bentoml.asgi_app(app)


class SDXLBase:
    
    def __init__(self) -> None:
        self.variant = 'fp16'
        if os.path.exists(config.CHECKPOINT_PATH):
            self.variant = None
        self.pipe = None
        self._init_pipeline()

    def _init_pipeline(self, ):
        torch.cuda.empty_cache()
        self.pipe = DiffusionPipeline.from_pretrained(config.CHECKPOINT_PATH, 
                                         torch_dtype=torch.float16, 
                                         use_safetensors=True, 
                                         variant=self.variant)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.to(device="cuda")


    @bentoml.api(route="/v1/images/generations")
    def txt2img(
            self,
            prompt: str,
            model: str,
            steps:  Union[int, None] = 20,
            seed: Union[int, None] = None,
            n: Union[int, None] = 1,
            height: Union[int, None] = 1024,
            width: Union[int, None] = 1024,
            negative_prompt: Union[str, None] = None,
    ) -> dict:
        try:
            req = ImageRequest(
                prompt=prompt,
                model=model,
                steps=steps,
                seed=seed,
                n=n,
                height=height,
                width=width,
                negative_prompt=negative_prompt
            )
        except Exception as e:
            raise UnprocessableEntity(message=str(e))
        try:
            response = handler_image_generation(self.pipe, req)
        except Exception as e:
            self._init_pipeline()
            raise InternalServerError(message='internal server error')
        return response.model_dump()
