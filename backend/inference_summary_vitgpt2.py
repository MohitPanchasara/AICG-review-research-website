# -*- coding: utf-8 -*-
import os, re
from typing import List, Dict, Any, Optional, Callable

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

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
    Fast, offline-capable summarizer using nlpconnect/vit-gpt2-image-captioning.
    - Loads from local dir if provided (VIT_LOCAL_DIR), else from HF id
    - Batches frames
    - Reuses ONE VideoCapture (seeks by msec)
    - Caps total segments with even spacing (SEGMENT_MAX)
    Returns {"items": [[start, end, caption], ...], "duration": float}
    """

    def __init__(
        self,
        model_id: str = "nlpconnect/vit-gpt2-image-captioning",
        local_dir: Optional[str] = None,
        device: Optional[str] = None,
        # generation (tunable by env)
        max_length: int = 60,
        min_length: int = 10,
        num_beams: int = 1,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 2,
        # performance
        batch_size: int = 4,
        use_fp16: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.use_fp16 = bool(use_fp16 and self.device == "cuda")

        load_from = local_dir if (local_dir and os.path.isdir(local_dir)) else model_id
        kwargs = {"local_files_only": bool(local_dir and os.path.isdir(local_dir))}

        self.model = VisionEncoderDecoderModel.from_pretrained(load_from, **kwargs)
        self.feat  = ViTImageProcessor.from_pretrained(load_from, **kwargs)
        self.tok   = AutoTokenizer.from_pretrained(load_from, **kwargs)

        self.model.to(self.device)
        if self.use_fp16: self.model.half()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # generation knobs (fast defaults)
        self.gen_kwargs = dict(
            max_length=int(max_length),
            min_length=int(min_length),
            num_beams=int(num_beams),
            length_penalty=float(length_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
        )

        # tame CPU thread thrash
        if self.device == "cpu":
            try:
                torch.set_num_threads(max(1, os.cpu_count() // 2))
            except Exception:
                pass

    def _video_meta(self, path: str):
        cap = cv2.VideoCapture(path)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = (total / fps) if (fps > 0 and total > 0) else 0.0
        cap.release()
        return fps, total, dur

    def _batch_caption(self, pil_list: List[Image.Image]) -> List[str]:
        if not pil_list: return []
        pixel_values = self.feat(images=pil_list, return_tensors="pt").pixel_values
        if self.use_fp16: pixel_values = pixel_values.half()
        pixel_values = pixel_values.to(self.device, non_blocking=True)

        with torch.no_grad():
            out_ids = self.model.generate(pixel_values=pixel_values, **self.gen_kwargs)
        texts = self.tok.batch_decode(out_ids, skip_special_tokens=True)
        return [_sanitize(t) for t in texts]

    @torch.inference_mode()
    def predict(
        self,
        video_path: str,
        segment_len: float = 3.0,
        stride: float = 1.0,
        max_segments: int = 30,
        progress: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Cap total segments to max_segments. If the naive stride would exceed it,
        we evenly space windows across the video duration.
        """
        _, _, dur = self._video_meta(video_path)
        if dur <= 0: dur = 6.0

        # Build start times with a hard cap
        # naive count if we stepped by 'stride':
        if dur <= segment_len:
            starts = np.array([0.0], dtype=np.float32)
        else:
            naive_n = int(np.floor((dur - 1e-6) / max(1e-6, stride))) + 1
            n = min(max_segments, max(1, naive_n))
            # evenly spaced starts so that end = start+segment_len <= dur
            last_start = max(0.0, dur - segment_len)
            starts = np.linspace(0.0, last_start, num=n, dtype=np.float32)

        items: List[List[Any]] = []
        total = len(starts)
        if progress: progress(10, "summary:start")

        # One persistent VideoCapture
        cap = cv2.VideoCapture(video_path)

        # Collect PIL images in batches
        pil_buf: List[Image.Image] = []
        meta_buf: List[tuple] = []
        done = 0

        try:
            for s in starts:
                e = float(min(dur, float(s) + segment_len))
                mid = (float(s) + e) / 2.0

                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, mid) * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    # skip silently
                    done += 1
                    if progress:
                        pct = 10 + int(50.0 * (done / max(1, total)))
                        progress(min(60, pct), f"summary:{done}")
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_buf.append(Image.fromarray(rgb))
                meta_buf.append((float(s), float(e)))

                if len(pil_buf) >= self.batch_size:
                    caps = self._batch_caption(pil_buf)
                    for (ss, ee), txt in zip(meta_buf, caps):
                        items.append([round(ss, 2), round(ee, 2), txt])
                    done += len(pil_buf)
                    pil_buf.clear(); meta_buf.clear()

                    if progress:
                        pct = 10 + int(50.0 * (done / max(1, total)))
                        progress(min(60, pct), f"summary:{done}")

            # flush tail
            if pil_buf:
                caps = self._batch_caption(pil_buf)
                for (ss, ee), txt in zip(meta_buf, caps):
                    items.append([round(ss, 2), round(ee, 2), txt])
                done += len(pil_buf)
                if progress:
                    pct = 10 + int(50.0 * (done / max(1, total)))
                    progress(min(60, pct), f"summary:{done}")

        finally:
            cap.release()

        if progress: progress(60, "summary:done")
        return {"items": items, "duration": round(float(dur), 2)}
