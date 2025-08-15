# -*- coding: utf-8 -*-
import os, re, time
from typing import List, Dict, Any, Optional, Callable

import cv2
import numpy as np
import torch
from PIL import Image

from transformers import (
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)

def _sanitize(text: str) -> str:
    text = text.replace("�", " ").replace("▁", " ")
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    if text and text[-1].isalnum():
        text += "."
    return text

class VideoSummarizer:
    """
    Offline+fast summarizer:
      - loads vit-gpt2 from local dir if available
      - batches frames to speed up generation
      - optional fp16 on GPU
      - returns {"items": [[start,end,caption],...], "duration": float}
    """

    def __init__(
        self,
        model_id: str = "nlpconnect/vit-gpt2-image-captioning",
        local_dir: Optional[str] = None,
        device: Optional[str] = None,
        # generation defaults = your original, but tunable via env
        max_length: int = 80,
        min_length: int = 22,
        num_beams: int = 4,
        length_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
        # performance
        batch_size: int = 8,
        use_fp16: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.use_fp16 = bool(use_fp16 and (self.device == "cuda"))

        # --- Load offline if folder exists; otherwise standard id
        load_from = local_dir if (local_dir and os.path.isdir(local_dir)) else model_id
        local_only = bool(local_dir and os.path.isdir(local_dir))
        kwargs = {"local_files_only": local_only}

        self.model = VisionEncoderDecoderModel.from_pretrained(load_from, **kwargs)
        self.feat  = ViTImageProcessor.from_pretrained(load_from, **kwargs)
        self.tok   = AutoTokenizer.from_pretrained(load_from, **kwargs)

        # fp16 on GPU (speeds up a lot)
        self.model.to(self.device)
        if self.use_fp16:
            self.model.half()

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # generation knobs (your original values)
        self.gen_kwargs = dict(
            max_length=int(max_length),
            min_length=int(min_length),
            num_beams=int(num_beams),
            length_penalty=float(length_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
        )

        if self.device == "cpu":
            torch.set_num_threads(max(1, os.cpu_count() // 2))

    def _video_meta(self, path: str):
        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = (total / fps) if (fps > 0 and total > 0) else 0.0
        cap.release()
        return fps, total, dur

    def _frame_at_time(self, path: str, t: float):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t) * 1000.0)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read frame at t={t:.2f}s")
        return frame

    @torch.inference_mode()
    def _batch_caption(self, pil_list: List[Image.Image]) -> List[str]:
        if not pil_list:
            return []
        pixel_values = self.feat(images=pil_list, return_tensors="pt").pixel_values
        # fp16 cast if needed
        if self.use_fp16:
            pixel_values = pixel_values.half()
        pixel_values = pixel_values.to(self.device, non_blocking=True)

        # autocast on CUDA keeps ViT in fp16 too
        cm = torch.cuda.amp.autocast if (self.device == "cuda" and self.use_fp16) else torch.cpu.amp.autocast
        with cm(enabled=(self.device == "cuda" and self.use_fp16)):
            out_ids = self.model.generate(pixel_values=pixel_values, **self.gen_kwargs)

        texts = self.tok.batch_decode(out_ids, skip_special_tokens=True)
        return [_sanitize(t) for t in texts]

    @torch.inference_mode()
    def predict(
        self,
        video_path: str,
        segment_len: float = 3.0,
        stride: float = 1.0,
        max_segments: int = 9999,
        progress: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        _, _, dur = self._video_meta(video_path)
        if dur <= 0:
            dur = 6.0

        # build segment starts
        starts: List[float] = []
        t = 0.0
        while t < max(0.0, dur - 1e-6):
            starts.append(t)
            t += stride
            if len(starts) >= max_segments:
                break

        items: List[List[Any]] = []
        n = len(starts)
        if progress: progress(10, "summary:start")

        # prefetch all center frames → PIL list
        centers = []
        for s in starts:
            e = float(min(dur, s + segment_len))
            centers.append(((s, e), (s + e) / 2.0))

        pil_buffer: List[Image.Image] = []
        meta_buffer: List[tuple] = []  # [(start,end), ...]

        processed = 0
        for (se, mid) in centers:
            bgr = self._frame_at_time(video_path, mid)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_buffer.append(Image.fromarray(rgb))
            meta_buffer.append(se)

            # flush batch
            if len(pil_buffer) >= self.batch_size:
                captions = self._batch_caption(pil_buffer)
                for (s, e), cap in zip(meta_buffer, captions):
                    items.append([round(float(s), 2), round(float(e), 2), cap])
                processed += len(pil_buffer)
                pil_buffer.clear(); meta_buffer.clear()

                if progress:
                    pct = 10 + int(50.0 * (processed / max(1, n)))
                    progress(min(60, pct), f"summary:{processed}")

        # tail
        if pil_buffer:
            captions = self._batch_caption(pil_buffer)
            for (s, e), cap in zip(meta_buffer, captions):
                items.append([round(float(s), 2), round(float(e), 2), cap])
            processed += len(pil_buffer)
            if progress:
                pct = 10 + int(50.0 * (processed / max(1, n)))
                progress(min(60, pct), f"summary:{processed}")

        if progress: progress(60, "summary:done")
        return {"items": items, "duration": round(float(dur), 2)}
