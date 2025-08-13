# AI Video Detector

Live demo: [https://ai-video-detector.vercel.app/](https://ai-video-detector.vercel.app/)

A lightweight, production-ready demo that classifies videos as AI-generated vs. real and visualizes results on a sleek, cyberpunk-inspired single-page UI.

## Initial Dataset Used:
[Kaggle](https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset/data)

## Model Details
- Backbone: `torchvision.models.densenet169`
- Head: Dropout(0.3) → Linear(in_features, 2)
- Training: CrossEntropy over 2 classes
- Inference: softmax → P(class=AI) = final model score
- Preprocessing: Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet mean/std)
- Frame Sampling: ~16 evenly spaced frames; mean probability over frames

## Highlights
- Single-video upload with inline preview
- DenseNet-169 backend (FastAPI + PyTorch) returning a single final model score (probability of AI)
- Real progress bar (load → sample → preprocess → infer → done)
- Mock mode for instant demos when no backend URL is set
- Expandable results layout — sections for summary, anomalies, frames, clips already scaffolded
- UI with glass panels & neon accents (Next.js + CSS Modules, no Tailwind)
- CORS-configurable backend, resilient error handling
- Hosting: Vercel (frontend) + Hugging Face Spaces (backend, CPU)

## Architecture
### Frontend (Next.js)
- [Browser / Next.js (Vercel)]
  - `POST /analyze` (video file)
  - `GET /status` (poll progress)
  - `GET /result` (final payload)
- [FastAPI (Spaces)]
  - DenseNet-169 (.pth)
  - final_model_score ∈ [0..1]

### Repo Structure (frontend)
```
pages/
  _app.js
  document.js
  index.js
  api/hello.js
components/
  HeaderBar.js
  WipDisclaimer.js
  UploadPanel.js
  ProcessingStatus.js
  ResultSection.js
  IntuitiveMeter.js
  AnomalyList.js
  DecisionCard.js
  ClipGrid.js
  FrameGrid.js
  FooterBar.js
  useVideoAnalysis.js
utils.js
styles/
  globals.css
  home.module.css
public/
  placeholder/* (optional mock assets)
```

### API (contract)
- `POST /analyze`
  - Body: multipart/form-data with field `file` (the video)
  - Response: `{ "job_id": "abc123def456" }`
- `GET /status?job_id=...`
  - Response (progressive):
    ```
    {
      "status": "queued|running|done|failed",
      "progress": 0,
      "partial": {
        "threshold_score": 0.5,
        "final_model_score": 0.82
      },
      "assets": [],
      "error": null
    }
    ```
- `GET /result?job_id=...`
  - Response: Same shape as `status`, finalized with `"status": "done"` (and may include timings).

## Local Development
You can run the frontend and backend independently. The frontend reads one env var: `NEXT_PUBLIC_API_BASE_URL`.

### 1) Frontend (Next.js)
```bash
# from project root
npm install
# point to your backend (local or Space)
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8080" > .env.local
npm run dev
# open http://localhost:3000
```
- Mock mode: If `NEXT_PUBLIC_API_BASE_URL` is unset, the UI simulates progress/results for quick demos.

### 2) Backend (FastAPI + PyTorch)
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# set env vars for this shell
$env:MODEL_PATH = "weights/model.pth"
$env:ALLOWED_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"
$env:UPLOAD_DIR = "C:\Windows\Temp\aiVideo" # optional on Windows
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## Deployment
### Frontend → Vercel
1. Import the frontend GitHub repo into Vercel.
2. Environment Variable (add for Production, Preview, Development):
   - `NEXT_PUBLIC_API_BASE_URL = https://<your-space>.hf.space`
3. Click Deploy. Your URL will be like `https://<your-project>.vercel.app`.

### Backend → Hugging Face Spaces (Docker)
1. Create a new Space → Docker → CPU Basic (free).
2. Add these files from `backend/`:
   - `Dockerfile` (see below)
   - `requirements.txt`
   - `main.py`, `inference.py`
   - `weights/model.pth` (upload directly or via Git LFS)
3. Set Space variables (Settings → Variables & secrets):
   - `MODEL_PATH=weights/model.pth`
   - `ALLOWED_ORIGINS=https://<your-project>.vercel.app,http://localhost:3000,http://127.0.0.1:3000`
   - (optional) `UPLOAD_DIR=/tmp/aiVideo` (code already defaults to this)
4. Deploy; wait until status is Running.
5. Test: `https://<your-space>.hf.space/health` → `{"ok": true}`.

#### Minimal Dockerfile:
```Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "main.py"]
```

## WIP Disclaimer
Work-in-progress alert: This is our smallest baseline model (DenseNet-169) and it's currently overfitting like it's cramming for finals. The real, multi-modal, multi-model architecture (smarter summaries, anomaly timelines, person checks, the works) is in the oven. In the meantime, upload a clip, kick the tires, and roast our baseline. Big upgrade landing soon. ✨

## CORS & Security
- Backend allows only origins from `ALLOWED_ORIGINS`.
- Add your Vercel domain (and any custom domain) there, then restart the Space.
- Keep test videos small (<100 MB) for faster CPU inference.
