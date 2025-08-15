# backend/Dockerfile
FROM python:3.10-slim

# System deps for OpenCV + video
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Writable upload dir (Spaces repo FS is read-only; /tmp is writable)
RUN mkdir -p /tmp/aivideo && chmod 777 /tmp/aivideo

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache vit-gpt2 into the image (OFFLINE at runtime). No heredoc used.
RUN python -c "from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer; \
mid='nlpconnect/vit-gpt2-image-captioning'; out='/app/weights/vit_captioner'; \
print('Caching model to', out); \
VisionEncoderDecoderModel.from_pretrained(mid).save_pretrained(out); \
ViTImageProcessor.from_pretrained(mid).save_pretrained(out); \
AutoTokenizer.from_pretrained(mid).save_pretrained(out); \
print('Done')"

# App code + DenseNet weights
COPY . /app

# ---- Backend env defaults (override in Space Settings > Variables) ----
ENV MODEL_PATH=weights/model.pth
ENV UPLOAD_DIR=/tmp/aivideo

# Use the locally cached captioner to run fully offline at runtime
ENV VIT_LOCAL_DIR=/app/weights/vit_captioner
ENV VIT_BATCH=8
ENV VIT_FP16=0                # CPU on Spaces → keep 0

# Generation knobs (your defaults)
ENV VIT_MAX_LEN=80
ENV VIT_MIN_LEN=22
ENV VIT_BEAMS=4
ENV VIT_LP=1.1
ENV VIT_NGRAM=3

# Timeline windowing
ENV SEGMENT_LEN=3.0
ENV SEGMENT_STRIDE=1.0
ENV SEGMENT_MAX=9999

# CORS: update after first deploy to include your Vercel domain(s)
ENV ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000

# Start FastAPI (Spaces provides $PORT)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
