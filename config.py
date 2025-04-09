import os

AUTH_TOKEN = os.getenv('AUTH_TOKEN', '9RHy3iT5EiabxHEYsPgL9eZBQ4jtKxZmSC92um4T0FFURJatwJ6uuUgRg3c4JiBy')
CHECKPOINT_PATH = os.getenv('CHECKPOINT_PATH', 'stabilityai/stable-diffusion-xl-base-1.0')
MODEL_NAMES = ['stabilityai/stable-diffusion-xl-base-1.0', 'black-forest-labs/FLUX.1-dev']
FLUX_DEV_CHECKPOINT = os.getenv('FLUX_DEV_CHECKPOINT', './weights/flux1-dev-fp8.safetensors')
HF_TOKEN = os.getenv('HF_TOKEN', 'null')
