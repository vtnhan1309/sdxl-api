import logging
import time
from fastapi import FastAPI
from models import ImageRequest
import shared
from handlers import handler_image_generation


# Set up logging configuration to remove default uvicorn logs
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Custom log format
    handlers=[logging.StreamHandler()] 
    )
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create the FastAPI app instance
app = FastAPI()

# Define a route for the root URL
@app.get("/health")
def health_check():
    return {"message": "OK"}


@app.post("/v1/images/generations")
def generate_image(req: ImageRequest):
    start_func = time.time()
    try:
        response = handler_image_generation(shared.base_sdxl_pipeline, req)
    except Exception as e:
        logger.error(f'Error: {e}, duration: {time.time() - start_func}')
        raise e
    logger.info(f'Success, duration: {time.time() - start_func:.4f}')
    return response
