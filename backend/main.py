# main.py — two-stage pipeline:
# 1) Summary via nlpconnect/vit-gpt2-image-captioning (your exact params)
# 2) DenseNet-169 score
# Uses FastAPI lifespan to ensure models load

import os, time, uuid, shutil, tempfile, threading, logging
from typing import Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import VideoScorer                 # your DenseNet-169 scorer
from inference_summary_vitgpt2 import VideoSummarizer  # NEW

# ---------- Env / config ----------
def _env_list(name: str, default: str) -> list[str]:
    v = os.getenv(name, default)
    return [x.strip() for x in v.split(",") if x.strip()]

MODEL_PATH = os.getenv("MODEL_PATH", "weights/model.pth")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "/tmp/aivideo")
ALLOWED_ORIGINS = _env_list("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")

VIT_MODEL_ID   = os.getenv("VIT_MODEL_ID", "nlpconnect/vit-gpt2-image-captioning")
VIT_LOCAL_DIR  = os.getenv("VIT_LOCAL_DIR", "weights/vit_captioner")  # <= saved folder from step A
VIT_BATCH      = int(os.getenv("VIT_BATCH", "8"))
VIT_FP16       = os.getenv("VIT_FP16", "1") == "1"

VIT_MAX_LEN    = int(os.getenv("VIT_MAX_LEN", "80"))
VIT_MIN_LEN    = int(os.getenv("VIT_MIN_LEN", "22"))
VIT_BEAMS      = int(os.getenv("VIT_BEAMS", "4"))
VIT_LP         = float(os.getenv("VIT_LP", "1.1"))
VIT_NGRAM      = int(os.getenv("VIT_NGRAM", "3"))

# vit-gpt2 knobs (can be tweaked via env if needed)
SEGMENT_LEN   = float(os.getenv("SEGMENT_LEN", "3.0"))
SEGMENT_STRIDE= float(os.getenv("SEGMENT_STRIDE", "1.0"))
SEGMENT_MAX   = int(os.getenv("SEGMENT_MAX", "9999"))

os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aivideo")

JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()

# ---------- Lifespan ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")

        logger.info("Loading DenseNet scorer...")
        app.state.scorer = VideoScorer(MODEL_PATH)

        logger.info("Loading vit-gpt2 summarizer (nlpconnect/vit-gpt2-image-captioning)...")
        app.state.summarizer = VideoSummarizer(
    model_id=VIT_MODEL_ID,
    local_dir=VIT_LOCAL_DIR,
    batch_size=VIT_BATCH,
    use_fp16=VIT_FP16,
    max_length=VIT_MAX_LEN,
    min_length=VIT_MIN_LEN,
    num_beams=VIT_BEAMS,
    length_penalty=VIT_LP,
    no_repeat_ngram_size=VIT_NGRAM,
)

        logger.info("Models loaded successfully.")
    except Exception:
        logger.exception("Startup failed while loading models")
        raise
    yield
    # no special shutdown

app = FastAPI(title="AIVideoClassifier (summary + score)", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- helpers ----------
def _put_job(job_id: str, data: Dict[str, Any]) -> None:
    with JOBS_LOCK: JOBS[job_id] = data

def _get_job(job_id: str) -> Dict[str, Any] | None:
    with JOBS_LOCK: return JOBS.get(job_id)

def _update_job(job_id: str, patch: Dict[str, Any]) -> None:
    with JOBS_LOCK:
        if job_id in JOBS: JOBS[job_id].update(patch)

def _set_progress(job_id: str, pct: int) -> None:
    pct = max(0, min(100, int(pct)))
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id]["progress"] = max(JOBS[job_id]["progress"], pct)

def _stage_progress(job_id: str, start: int, end: int):
    span = max(1, end - start)
    def cb(pct: int, _phase: str):
        mapped = start + int((max(0, min(100, pct)) / 100.0) * span)
        _set_progress(job_id, mapped)
    return cb

# ---------- routes ----------
class AnalyzeResp(BaseModel):
    job_id: str

@app.get("/health")
def health():
    ok = hasattr(app.state, "scorer") and hasattr(app.state, "summarizer")
    return {"ok": bool(ok)}

@app.post("/analyze", response_model=AnalyzeResp)
def analyze(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    safe_suffix = os.path.splitext(file.filename or "")[-1][:8]
    with tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=safe_suffix, delete=False) as tmpf:
        shutil.copyfileobj(file.file, tmpf)
        dst_path = tmpf.name

    job = {
        "status": "queued",
        "progress": 0,
        "partial": {
            "threshold_score": 0.5,
            "summary_model": "nlpconnect/vit-gpt2-image-captioning",
            "summary_timeline": [],
            "summary_duration": None,
        },
        "assets": {},
        "error": None,
        "path": dst_path,
        "started_at": time.time(),
    }
    _put_job(job_id, job)
    threading.Thread(target=_worker, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}

def _worker(job_id: str) -> None:
    job = _get_job(job_id)
    if not job: return
    _update_job(job_id, {"status": "running", "progress": 1})

    summarizer: VideoSummarizer = app.state.summarizer
    scorer:     VideoScorer     = app.state.scorer

    try:
        # ---- Stage 1: Summary (10..60)
        sum_cb = _stage_progress(job_id, 10, 60)
        sres = summarizer.predict(
            job["path"],
            segment_len=SEGMENT_LEN,
            stride=SEGMENT_STRIDE,
            max_segments=SEGMENT_MAX,
            progress=sum_cb
        )
        _update_job(job_id, {
            "partial": {
                **job["partial"],
                "summary_timeline": sres.get("items", []),
                "summary_duration": sres.get("duration", None),
            }
        })
        _set_progress(job_id, 60)

        # ---- Stage 2: DenseNet score (60..100)
        cls_cb = _stage_progress(job_id, 60, 100)
        cres = scorer.predict(job["path"], progress=cls_cb)
        _update_job(job_id, {
            "partial": {
                **_get_job(job_id)["partial"],
                "final_model_score": float(cres.get("score", 0.0)),
            },
            "elapsed_s": float(cres.get("elapsed_s", 0.0)),
            "status": "done",
            "progress": 100,
        })

    except Exception as e:
        _update_job(job_id, {"status": "failed", "error": str(e)})
    finally:
        try: os.remove(job["path"])
        except Exception: pass

@app.get("/status")
def status(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return {
        "status": job["status"],
        "progress": int(job["progress"]),
        "partial": job["partial"],
        "assets": job["assets"],
        "error": job["error"],
    }

@app.get("/result")
def result(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job_id")
    if job["status"] != "done":
        return status(job_id)
    return {
        "status": job["status"],
        "progress": int(job["progress"]),
        "partial": job["partial"],
        "assets": job["assets"],
        "elapsed_s": job.get("elapsed_s", None),
        "error": job["error"],
    }
