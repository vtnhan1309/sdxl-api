import requests
import pytest
import http

URL = 'http://0.0.0.0:8000/v1/images/generations'
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

    def test_steps(self):
        response = self._generate_image(steps=-1)
        assert response.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY
        response = self._generate_image(steps=200)
        assert response.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY
    
    def test_prompt(self):
        response = self._generate_image(prompt='')
        assert response.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY

    def test_model(self):
        response = self._generate_image(model='abc')
        assert response.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY
        response = self._generate_image(model='')
        assert response.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY

    def test_seed(self):
        response = self._generate_image(seed=-1)
        assert response.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY

    def test_n(self):
        response = self._generate_image(n=-1)
        assert response.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY
        response = self._generate_image(n=5)
        assert response.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY

    def test_height_width(self):
        response = self._generate_image(height=-1)
        assert response.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY
        response = self._generate_image(height=3000)
        assert response.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY
        response = self._generate_image(width=-1)
        assert response.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY
        response = self._generate_image(width=3000)
        assert response.status_code == http.HTTPStatus.UNPROCESSABLE_ENTITY
