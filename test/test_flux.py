import requests
import http

URL = 'http://0.0.0.0:8000/v1/images/generations'
MODEL_NAME = 'black-forest-labs/FLUX.1-dev'

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

    def test_gen_1_image(self):
        response = self._generate_image(prompt='a cow', 
                                        steps=30)
        assert response.status_code == http.HTTPStatus.OK
        data = response.json()
        assert len(data['data']) == 1

    def test_gen_4_image(self):
        response = self._generate_image(prompt='a cow', 
                                        steps=30, n=4)
        assert response.status_code == http.HTTPStatus.OK
        data = response.json()
        assert len(data['data']) == 4

    def test_gen_1_image_512(self):
        response = self._generate_image(prompt='a cow', 
                                        steps=30, n=1, height=512, width=512)
        assert response.status_code == http.HTTPStatus.OK
        data = response.json()
        assert len(data['data']) == 1

    def test_full_param(self):
        response = self._generate_image(prompt='a cow', 
                                        negative_prompt='bad quality',
                                        height=1024, width=1024,
                                        steps=20, n=1, seed=None)
        assert response.status_code == http.HTTPStatus.OK
        data = response.json()
        assert len(data['data']) == 1
