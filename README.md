# Deepfake Video Detector

An AI system that takes a video, detects faces frame-by-frame with MTCNN, scores
each face with a deepfake classifier, and aggregates the scores into a
**FAKE / REAL** verdict with confidence, per-frame probabilities,
and suspicious time segments. Ships with a FastAPI backend and a single-file
demo web UI.

**FYP — BSCS, IBADAT International University Islamabad** · Student: Ammad Ullah (SAP ID 5112323040)

## ⚡ Quick demo — no training needed

A ready-to-run demo using a public pretrained classifier
(`king1oo1/deepfake-model`, SigLIP2, Apache 2.0 — see `docs/PRETRAINED_MODEL.md`):

```bash
pip install -r requirements.txt   # includes transformers/safetensors for the demo
python pretrained_demo.py --video samples/test_real.mp4
# → {"verdict": "REAL", "confidence": 0.7868, ...}
python pretrained_demo.py --video samples/test_fake.mp4
# → {"verdict": "FAKE", "confidence": 0.6988, ...}
```

Runs on CPU. The video pipeline (frame sampling, MTCNN face extraction,
temporal aggregation) is this project's own code; only the image classifier
weights are pretrained. Full honest details + measured results in
`docs/PRETRAINED_MODEL.md`.

---

> **About metrics:** this repository ships **no pretrained weights and no
> accuracy claims**. All metrics (`checkpoints/metrics.json`,
> `confusion_matrix.png`) are produced by *your own* training run — report those
> numbers in your FYP documentation, never numbers copied from elsewhere.

---

## 1. Setup

```bash
cd fyp-deepfake-detector
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# CPU-only install (recommended for the API/demo machine):
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Training needs a GPU — use the free Colab notebook (`notebooks/train_on_colab.ipynb`);
torch/torchvision are preinstalled there, so the notebook skips those two lines.

Verify the CLIs:
```bash
python -m src.data.download_data --help
python -m src.data.extract_frames --help
python -m src.data.extract_faces --help
python -m src.training.train --help
python -m src.inference.predict --help
```

## 2. Get data

```bash
python -m src.data.download_data --dataset all   # prints official sources + steps
# ... download FaceForensics++ and/or Celeb-DF, then organise:
#   data/real/<video>.mp4   (pristine)
#   data/fake/<video>.mp4   (manipulated)
python -m src.data.download_data --check         # verify the layout
```

## 3. Extract frames → faces → manifest

```bash
# frames at 2 fps, mirroring data/real+fake layout
python -m src.data.extract_frames --batch-dir data --out-dir data/frames --fps 2

# MTCNN face crops (224x224) + manifest CSV  (slow: run on Colab/GPU)
python -m src.data.extract_faces --frames-dir data/frames --out-dir data/faces \
    --manifest data/manifest.csv --label-from-parent
```

Tip: start with ~200 real + 200 fake videos to verify the pipeline end-to-end,
then scale up.

## 4. Train the model

On Colab (recommended): open `notebooks/train_on_colab.ipynb`
(**Runtime → Change runtime type → GPU**) and run all cells. It installs deps,
runs the three steps above, trains, and lets you download `checkpoints/best.pt`.

Locally (CPU works, but slow):
```bash
python -m src.training.train --config configs/config.yaml
# outputs: checkpoints/best.pt, checkpoints/metrics.json,
#          checkpoints/confusion_matrix.png
```

Training details: AdamW (`lr=3e-4`), cosine LR schedule, mixed precision on
CUDA, early stopping (patience 5 on val loss), best checkpoint by **validation
F1**. The train/val split is by **video**, never by frame, to avoid leakage.
All hyperparameters live in `configs/config.yaml`.

## 5. Predict on a video (CLI)

```bash
python -m src.inference.predict --video samples/clip.mp4 --out result.json
# prints: verdict, confidence, video_score, frames_analyzed,
#         suspicious_segments, per-frame scores
```

## 6. Run the demo API + web UI

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
# open http://127.0.0.1:8000  → upload a video, see the verdict card
```

- `POST /predict` — multipart upload (`file`, ≤ 50 MB) → verdict JSON.
- `GET /health` — `{"status":"ok","model_loaded":...}` (never loads the model).
- If `checkpoints/best.pt` is missing, `/predict` returns **503** with a clear
  "train a model first" message instead of crashing.

## Repository layout

```
configs/config.yaml        all hyperparameters + paths (single source of truth)
src/data/                  download guide, frame + face extraction
src/models/detector.py     EfficientNet-B0 + Dropout(0.3) + Linear(1280→1)
src/training/              manifest dataset (video-level split) + training loop
src/inference/             temporal aggregation + video prediction (+CLI)
src/api/main.py            FastAPI: /predict, /health, static demo UI
src/frontend/index.html    single-file demo UI (vanilla JS, no build step)
notebooks/                 Colab training notebook
docs/ARCHITECTURE.md       pipeline diagram, module map, API contract
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: checkpoints/best.pt` | Train first (step 4); copy `best.pt` into `checkpoints/`. |
| MTCNN very slow on CPU | Expected — face extraction is the bottleneck; use Colab GPU. |
| `num_workers` errors on Windows | Set `training.num_workers: 0` in `configs/config.yaml`. |
| CUDA out of memory | Lower `training.batch_size` to 16 or 8. |
| `NO_FACE_DETECTED` verdict | Video has no clear front-facing face; try another clip. |

## Viva cheat-sheet

- **Why top-50% mean?** Fake videos show artefacts only in some frames; averaging
  everything dilutes the signal. The top-half mean focuses on suspicious frames.
- **Why video-level split?** Frames from one video are near-duplicates; a frame
  split leaks them into validation and inflates accuracy.
- **Why EfficientNet-B0?** Smallest EfficientNet — trains on free Colab GPUs —
  with ImageNet pretraining that transfers well to face forensics.
- **Limitations (say these yourself):** needs a visible face; heavy compression
  hurts; unseen generators may evade it — hence cross-dataset evaluation.
