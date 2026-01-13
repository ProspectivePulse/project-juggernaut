# Use an official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install System Dependencies (GLIB, etc. required for OpenCV/Gym)
RUN apt-get update && apt-get install -y \
	libgl1 \
	libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*
	
# 2. Install Python Dependencies
# We copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the Source Code
COPY src/ src/
COPY main.py .

# 4. Set the Default Entrypoint
# This enables running: "docker run my-bot train" or "docker run my-bot server"
CMD ["python", "main.py", "api"]