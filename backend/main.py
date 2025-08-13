import os, uuid, shutil, threading, time
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import VideoScorer

# ---- Env/config ----
MODEL_PATH = os.getenv("MODEL_PATH", "weights/model.pth")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "tmp")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")]

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="DenseNet169 Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory job store (fine for a demo)
JOBS: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
def _startup():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"MODEL_PATH not found: {MODEL_PATH}")
    app.state.scorer = VideoScorer(MODEL_PATH)
    print(f"[startup] Model loaded from {MODEL_PATH}")


@app.get("/health")
def health():
    return {"ok": True}


class AnalyzeResp(BaseModel):
    job_id: str


@app.post("/analyze", response_model=AnalyzeResp)
def analyze(file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:12]
    dst_path = os.path.join(UPLOAD_DIR, f"{job_id}_{file.filename}")

    # save uploaded file
    with open(dst_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "partial": {
            # provide a threshold so the frontend can calculate a decision if needed
            "threshold_score": 0.5
        },
        "assets": {},
        "error": None,
        "path": dst_path,
        "started_at": time.time(),
    }

    threading.Thread(target=_worker, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


def _worker(job_id: str):
    job = JOBS[job_id]
    job["status"] = "running"
    scorer: VideoScorer = app.state.scorer

    def progress(pct: int, _phase: str):
        job["progress"] = int(max(0, min(100, pct)))

    try:
        res = scorer.predict(job["path"], progress=progress)
        # put only what the frontend needs for now
        job["partial"]["final_model_score"] = float(res["score"])
        job["elapsed_s"] = float(res["elapsed_s"])
        job["progress"] = 100
        job["status"] = "done"
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        # cleanup
        try:
            os.remove(job["path"])
        except Exception:
            pass


@app.get("/status")
def status(job_id: str):
    job = JOBS.get(job_id)
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
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown job_id")
    if job["status"] != "done":
        return status(job_id)
    return {
        "status": job["status"],
        "progress": int(job["progress"]),
        "partial": job["partial"],  # includes final_model_score + threshold_score
        "assets": job["assets"],
        "elapsed_s": job.get("elapsed_s", None),
        "error": job["error"],
    }
