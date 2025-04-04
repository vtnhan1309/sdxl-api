import requests
import pytest
import http

URL = 'http://0.0.0.0:8000/v1/images/generations'
HEALTH_URL = 'http://0.0.0.0:8000/health'
MODEL_NAME = 'stabilityai/stable-diffusion-xl-base-1.0'

class TestImageGenValidator:
    def _generate_image(self, prompt='a boy', negative_prompt='', n=1, 
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

    def test_health(self):
        respone = requests.get(HEALTH_URL)
        assert respone.status_code == http.HTTPStatus.OK
