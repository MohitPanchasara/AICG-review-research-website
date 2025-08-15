# backend/tools/cache_vitgpt2.py
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"
OUT_DIR  = "weights/vit_captioner"  # local folder to load from later

print("Downloading & caching model to", OUT_DIR)
VisionEncoderDecoderModel.from_pretrained(MODEL_ID).save_pretrained(OUT_DIR)
ViTImageProcessor.from_pretrained(MODEL_ID).save_pretrained(OUT_DIR)
AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(OUT_DIR)
print("Done.")
