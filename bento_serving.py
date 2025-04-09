import logging
import os
from fastapi import FastAPI
import bentoml
from diffusers import DiffusionPipeline, FluxPipeline
import torch
from bentoml.exceptions import UnprocessableEntity, InternalServerError
from typing_extensions import Union
import handlers
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


class ImageGenService:
    
    def __init__(self) -> None:
        self.sdxl_pipeline = None
        self.flux_pipeline = None
        self._init_sdxl_pipeline()
        self._init_flux_dev_pipeline()
        self._init_logger()

    def _init_logger(self):
        # Create a stream handler
        ch = logging.StreamHandler()

        # Set a format for the handler
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Get the BentoML logger
        self.bentoml_logger = logging.getLogger("bentoml")

        # Add the handler to the BentoML logger
        self.bentoml_logger.addHandler(ch)

        # Set the desired logging level (e.g., DEBUG)
        self.bentoml_logger.setLevel(logging.DEBUG)


    def _init_sdxl_pipeline(self, ):
        torch.cuda.empty_cache()
        self.sdxl_pipeline = DiffusionPipeline.from_pretrained(config.CHECKPOINT_PATH, 
                                         torch_dtype=torch.float16, 
                                         use_safetensors=True)
        self.sdxl_pipeline.set_progress_bar_config(disable=True)
        self.sdxl_pipeline.to(device="cuda")

    def _init_flux_dev_pipeline(self, ):
        torch.cuda.empty_cache()
        self.flux_pipeline = FluxPipeline.from_single_file(config.FLUX_DEV_CHECKPOINT, 
                                         torch_dtype=torch.bfloat16, 
                                         use_safetensors=True, 
                                         token=config.HF_TOKEN)
        self.flux_pipeline.set_progress_bar_config(disable=True)
        self.flux_pipeline.to(device="cuda")

    def _log_request(self, **kwargs):
        log = []
        for k, v in kwargs.items():
            log.append(f'{k}: {v}')
        self.bentoml_logger.info(', '.join(log))

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
        self._log_request(**req.model_dump())
        try:
            if req.model == config.MODEL_NAMES[0]:
                response = handlers.handler_image_generation_sdxl(self.sdxl_pipeline, 
                                                                  req)
            elif req.model == config.MODEL_NAMES[1]:
                response = handlers.handler_image_generation_flux(self.flux_pipeline, 
                                                                  req)
        except Exception as e:
            if req.model == config.MODEL_NAMES[0]:
                self._init_sdxl_pipeline()
            elif req.model == config.MODEL_NAMES[1]:
                self._init_flux_dev_pipeline()
            raise InternalServerError(message='internal server error')
        return response.model_dump()
