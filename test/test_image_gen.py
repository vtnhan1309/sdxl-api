import requests
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

    def test_gen_1_image_768(self):
        response = self._generate_image(prompt='a cow', 
                                        steps=30, n=1, height=768, width=768)
        assert response.status_code == http.HTTPStatus.OK
        data = response.json()
        assert len(data['data']) == 1

    def test_seed(self):
        response = self._generate_image(prompt='a cow', 
                                        steps=20, n=1, seed=None)
        assert response.status_code == http.HTTPStatus.OK
        data = response.json()
        image1 = data['data'][0]['b64_json']
        response = self._generate_image(prompt='a cow', 
                                        steps=20, n=1, seed=None)
        assert response.status_code == http.HTTPStatus.OK
        data = response.json()
        image2 = data['data'][0]['b64_json']
        assert image1 != image2
        response = self._generate_image(prompt='a cow', 
                                        steps=20, n=1, seed=2)
        assert response.status_code == http.HTTPStatus.OK
        data = response.json()
        image3 = data['data'][0]['b64_json']
        response = self._generate_image(prompt='a cow', 
                                        steps=20, n=1, seed=2)
        assert response.status_code == http.HTTPStatus.OK
        data = response.json()
        image4 = data['data'][0]['b64_json']
        assert image3 == image4

    def test_full_param(self):
        response = self._generate_image(prompt='a cow', 
                                        negative_prompt='bad quality',
                                        height=768, width=768,
                                        steps=20, n=1, seed=None)
        assert response.status_code == http.HTTPStatus.OK
        data = response.json()
        assert len(data['data']) == 1
