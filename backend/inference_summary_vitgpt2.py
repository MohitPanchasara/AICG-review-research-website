# -*- coding: utf-8 -*-
import os, re, time
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
    vit-gpt2 summary with sequential decode + batch captioning.
    Now supports a hard time budget so it never runs forever on CPU.
    """

    def __init__(
        self,
        model_id: str = "nlpconnect/vit-gpt2-image-captioning",
        local_dir: Optional[str] = None,
        device: Optional[str] = None,
        # CPU-friendly generation defaults
        max_length: int = 48,
        min_length: int = 8,
        num_beams: int = 1,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 2,
        # performance
        batch_size: int = 3,
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

        self.gen_kwargs = dict(
            max_length=int(max_length),
            min_length=int(min_length),
            num_beams=int(num_beams),
            length_penalty=float(length_penalty),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
        )

        if self.device == "cpu":
            try: torch.set_num_threads(max(1, os.cpu_count() // 2))
            except Exception: pass

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
        max_segments: int = 12,         # ↓ lower cap for prod CPU
        time_budget_s: float = 25.0,    # ← hard ceiling per summary stage
        progress: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        fps, total, dur = self._video_meta(video_path)
        if dur <= 0 or fps <= 0:
            fps, dur, total = 30.0, 6.0, int(30 * 6.0)

        # Build evenly spaced windows with a hard cap
        if dur <= segment_len:
            starts = np.array([0.0], dtype=np.float32)
        else:
            naive_n = int(np.floor((dur - 1e-6) / max(1e-6, stride))) + 1
            n = min(max_segments, max(1, naive_n))
            last_start = max(0.0, dur - segment_len)
            starts = np.linspace(0.0, last_start, num=n, dtype=np.float32)

        targets, metas = [], []
        for s in starts:
            e = float(min(dur, float(s) + segment_len))
            targets.append((float(s) + e) / 2.0)
            metas.append((float(s), float(e)))
        targets = np.array(targets, dtype=np.float32)
        total_targets = len(targets)

        if progress: progress(10, "summary:start")

        cap = cv2.VideoCapture(video_path)
        pil_buf: List[Image.Image] = []
        meta_buf: List[tuple] = []
        items: List[List[Any]] = []

        next_k = 0
        processed = 0
        frame_idx = 0
        t0 = time.perf_counter()

        try:
            while next_k < total_targets:
                # time budget guard
                if (time.perf_counter() - t0) > time_budget_s:
                    break

                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                t = frame_idx / fps

                while next_k < total_targets and t >= targets[next_k]:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_buf.append(Image.fromarray(rgb))
                    meta_buf.append(metas[next_k])
                    next_k += 1

                    if len(pil_buf) >= self.batch_size:
                        caps = self._batch_caption(pil_buf)
                        for (ss, ee), txt in zip(meta_buf, caps):
                            items.append([round(ss, 2), round(ee, 2), txt])
                        processed += len(pil_buf)
                        pil_buf.clear(); meta_buf.clear()

                        if progress:
                            pct = 10 + int(50.0 * (processed / max(1, total_targets)))
                            progress(min(60, pct), f"summary:{processed}")

                frame_idx += 1

            # flush tail (if time budget still allows)
            if pil_buf and (time.perf_counter() - t0) <= time_budget_s:
                caps = self._batch_caption(pil_buf)
                for (ss, ee), txt in zip(meta_buf, caps):
                    items.append([round(ss, 2), round(ee, 2), txt])
                processed += len(pil_buf)
                if progress:
                    pct = 10 + int(50.0 * (processed / max(1, total_targets)))
                    progress(min(60, pct), f"summary:{processed}")
        finally:
            cap.release()

        if progress: progress(60, "summary:done")
        return {"items": items, "duration": round(float(dur), 2)}
