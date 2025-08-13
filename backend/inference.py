import os, time, uuid
from typing import Callable, Optional, Dict, Any, List

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ---- Model defined exactly like your training code ----
class DenseNetVideo(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        backbone = models.densenet169(weights=None)  # no download; we load your weights
        in_f = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


class VideoScorer:
    """
    Loads your DenseNet169 and returns a single probability score for class-1 (FAKE/AI).
    """
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, weights_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DenseNetVideo(num_classes=2)
        self._load_weights(weights_path)
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # Match training preprocessing
        self.tfm = transforms.Compose([
            transforms.Resize(256, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
        ])

    def _load_weights(self, weights_path: str):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights not found: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu")
        # Support both formats:
        state = ckpt.get("model_state") if isinstance(ckpt, dict) else None
        if state is None:
            # maybe already a raw state_dict
            state = ckpt if isinstance(ckpt, dict) else None
        if state is None:
            raise RuntimeError("Unsupported checkpoint format. Expected dict with 'model_state' or raw state_dict.")

        # strip common prefixes like 'module.' or 'model.'
        fixed = {}
        for k, v in state.items():
            nk = k
            for pref in ("module.", "model."):
                if nk.startswith(pref):
                    nk = nk[len(pref):]
            fixed[nk] = v
        missing, unexpected = self.model.load_state_dict(fixed, strict=False)
        if missing:
            print(f"[warn] missing keys: {missing}")
        if unexpected:
            print(f"[warn] unexpected keys: {unexpected}")

    def _evenly_sample_frame_indices(self, total_frames: int, max_frames: int = 16) -> List[int]:
        if total_frames <= 0:
            return []
        n = min(max_frames, total_frames)
        # linspace across the whole video
        idxs = np.linspace(0, total_frames - 1, num=n, dtype=int).tolist()
        # unique & sorted
        return sorted(set(idxs))

    def _read_frames(self, video_path: str, indices: List[int]) -> List[np.ndarray]:
        if not indices:
            return []
        cap = cv2.VideoCapture(video_path)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frm = cap.read()
            if ok and frm is not None:
                frames.append(frm)  # BGR
        cap.release()
        return frames

    def _total_frames(self, video_path: str) -> int:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        # fallback if 0
        if total <= 0:
            cnt = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                cnt += 1
            total = cnt
        cap.release()
        return total

    def _preprocess_batch(self, frames_bgr: List[np.ndarray]) -> torch.Tensor:
        if not frames_bgr:
            return torch.empty(0, 3, 224, 224)
        ten = []
        for bgr in frames_bgr:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            ten.append(self.tfm(img))
        return torch.stack(ten, dim=0)

    @torch.inference_mode()
    def predict(self, video_path: str, progress: Optional[Callable[[int, str], None]] = None) -> Dict[str, Any]:
        """
        Returns:
          {
            "score": float,           # probability of class-1 (FAKE)
            "frames_used": int,
            "total_frames": int,
            "elapsed_s": float
          }
        """
        t0 = time.perf_counter()
        if progress: progress(5,  "init")

        total = self._total_frames(video_path)
        idxs = self._evenly_sample_frame_indices(total, max_frames=16)
        if progress: progress(20, "sampled")

        frames = self._read_frames(video_path, idxs)
        if progress: progress(40, "loaded_frames")

        batch = self._preprocess_batch(frames).to(self.device, non_blocking=True)
        if progress: progress(60, "preprocessed")

        if batch.shape[0] == 0:
            prob1 = 0.5  # neutral fallback
        else:
            logits_list = []
            bs = 8
            for i in range(0, batch.shape[0], bs):
                out = self.model(batch[i:i+bs])         # [B,2]
                logits_list.append(out)
            logits = torch.cat(logits_list, dim=0)       # [N,2]
            probs = torch.softmax(logits, dim=1)[:, 1]   # p(class=1)
            prob1 = float(probs.mean().item())

        if progress: progress(95, "inferred")
        elapsed = time.perf_counter() - t0
        if progress: progress(100, "done")

        return {
            "score": prob1,
            "frames_used": int(len(frames)),
            "total_frames": int(total),
            "elapsed_s": float(elapsed),
        }
