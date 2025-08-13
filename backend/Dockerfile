# Dockerfile (for Hugging Face Spaces)
FROM python:3.10-slim

# system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# AFTER apt-get block
RUN mkdir -p /tmp/aivideo && chmod 777 /tmp/aivideo

WORKDIR /app

# requirements first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# copy code and weights
COPY . /app

# Hugging Face Spaces sets $PORT for you
ENV PORT=7860
EXPOSE 7860

# Let FastAPI see where the model is
ENV MODEL_PATH=weights/model.pth
# Allow your local + vercel domain (you can add more later)
ENV ALLOWED_ORIGINS=https://your-project.vercel.app,http://localhost:3000,http://127.0.0.1:3000

# Start API
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
