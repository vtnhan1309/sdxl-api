# Image generation API example
## Setup
### Build image
```
docker build -t imagegen:v0.0.1 .
```
### Start service
```
docker-compose -p imagegen -f docker-compose.yml up -d
```
## Test pipeline
### SDXL base 1.0
```
curl --location 'http:/0.0.0.0:8000/v1/images/generations' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "A hyper-realistic portrait of a young woman with long brown hair, wearing a white sweater, with natural soft lighting, a blurred background, and a warm, cozy atmosphere. Her skin is flawless, and her expression is calm and serene. The image has high-definition details and lifelike textures.",
    "negative_prompt": "bad quality",
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "steps": 30,
    "n": 1,
    "height": 1024,
    "width": 1024
}'
```
### Flux1 dev
```
curl --location 'http:/0.0.0.0:8000/v1/images/generations' \
--header 'Content-Type: application/json' \
--data '{
    "prompt": "a deep-sea dive into a sunken pirate ship surrounded by sharks",
    "negative_prompt": "bad quality",
    "model": "black-forest-labs/FLUX.1-dev",
    "steps": 30,
    "n": 1,
    "height": 1024,
    "width": 1024
}'
```
