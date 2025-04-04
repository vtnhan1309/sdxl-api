import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '1m', target: 10 },
  ],
};

export default function () {
  // Sample GET request to the /health endpoint
  // let res = http.get('http://localhost:8000/health');
  // check(res, {
  //   'health check status is 200': (r) => r.status === 200,
  // });

  // Sample POST request to the /items endpoint
  const payload = JSON.stringify({
    prompt: 'a boy',
    negative_prompt: 'bad quality',
    model: 'stabilityai/stable-diffusion-xl-base-1.0',
    steps: 30,
    n: 4,
    height: 512,
    width: 512
  });

  const params = {
    headers: { 'Content-Type': 'application/json' },
  };

  let res = http.post('http://localhost:8000/v1/images/generations', payload, params);
  check(res, {
    'create item status is 200': (r) => r.status === 200,
  });

  sleep(3); // Simulate user thinking
}

