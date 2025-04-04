# start fastapi
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4

# start bentoml serving
bentoml serve bento_serving.py --host 0.0.0.0 --port 8000

# start loadtest
~/k6-v0.43.0-linux-amd64/k6 run loadtest/loadtest.js
