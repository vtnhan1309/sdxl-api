import requests
import http
import base64
import io
from PIL import Image


URL = 'http://0.0.0.0:8000/v1/images/generations'
MODEL_NAME = 'stabilityai/stable-diffusion-xl-base-1.0'


def _generate_image(prompt='a boy', negative_prompt='', n=1, 
                    model=MODEL_NAME, steps=1, seed=1, height=1024, 
                    width=1024):
    payload = {
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'model': model,
        'steps': steps,
        'seed': seed,
        'n': n,
        'height': height,
        'width': width
    }
    response = requests.post(URL, json=payload)
    return response


response = _generate_image(prompt='Astronaut in a jungle, cold color palette, muted colors, detailed, 8k', 
                           negative_prompt='bad quality',
                                steps=30)
assert response.status_code == http.HTTPStatus.OK
data = response.json()
assert len(data['data']) == 1
image_raw = data['data'][0]['b64_json']
# Decode the base64 string
image_data = base64.b64decode(image_raw)

# Convert to a Pillow image
image = Image.open(io.BytesIO(image_data))

# Optionally save the image to a file
image.save('output_image.png')
