# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install dependencies from the requirements.txt file
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that FastAPI will run on
EXPOSE 8000

# Copy the rest of your FastAPI application into the container
COPY . .

# Command to run FastAPI with uvicorn
# CMD ["bentoml", "serve", "bento_serving.py" ,"--host", "0.0.0.0", "--port", "8000"]
