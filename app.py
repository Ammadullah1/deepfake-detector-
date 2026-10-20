"""Deepfake Video Detector — Gradio demo for Hugging Face Spaces.

No terminal needed: upload this whole project folder to a Hugging Face Space
(Gradio SDK) via the website, upload your trained checkpoints/best.pt the same
way, and you get a public link to demo your FYP.

Pipeline (reused from src/): video -> frames (2 fps) -> MTCNN face crops ->
EfficientNet-B0 per-frame fake probabilities -> temporal aggregation ->
FAKE / REAL verdict with confidence and suspicious segments.
"""

import sys
import traceback
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.inference.predict import (  # noqa: E402
    MODEL_NOT_TRAINED_MSG,
    load_pipeline,
    predict_video,
)

_pipeline = None
_pipeline_error = None


def get_pipeline():
    """Load the model lazily on first request (keeps Space startup fast)."""
    global _pipeline, _pipeline_error
    if _pipeline is None and _pipeline_error is None:
        try:
            _pipeline = load_pipeline(
                config_path=str(ROOT / "configs" / "config.yaml"))
        except FileNotFoundError as exc:
            _pipeline_error = str(exc)
        except Exception:  # noqa: BLE001 - show the real error in the UI
            _pipeline_error = traceback.format_exc(limit=3)
    return _pipeline


def _format_result(result: dict) -> str:
    verdict = result.get("verdict", "UNKNOWN")
    conf = result.get("confidence", 0.0) * 100
    frames = result.get("frames_analyzed", 0)

    if verdict == "NO_FACE_DETECTED":
        return (
            "## ⚪ No face detected\n\n"
            f"{result.get('note', '')}\n\n"
            "Try a video with a visible, front-facing person."
        )

    badge = "🔴 **FAKE** — this video looks manipulated" if verdict == "FAKE" \
        else "🟢 **REAL** — no manipulation detected"
    bars = "█" * int(conf // 5) + "░" * (20 - int(conf // 5))

    md = f"## {badge}\n\n**Confidence:** {conf:.1f}%\n`{bars}`\n\n"
    md += f"Frames analyzed: **{frames}**\n\n"

    segments = result.get("suspicious_segments", []) or []
    if segments:
        md += "### ⚠️ Suspicious segments\n"
        for seg in segments:
            md += (f"- {seg['start_sec']:.1f}s → {seg['end_sec']:.1f}s "
                   f"(peak score {seg.get('peak_score', 0):.2f})\n")
    else:
        md += "No suspicious segments found.\n"
    return md


def detect(video_path):
    if not video_path:
        return "Please upload a video first."
    pipe = get_pipeline()
    if pipe is None:
        return (
            "## ⚠️ Model not trained yet\n\n"
            "This Space needs your trained weights before it can detect anything.\n\n"
            "**What to do (all in the browser):**\n"
            "1. Train the model with `notebooks/train_on_colab.ipynb` on Google Colab (free GPU).\n"
            "2. Download `checkpoints/best.pt` from Colab.\n"
            "3. In this Space: **Files** tab → **Add file** → **Upload** → "
            "create the folder path `checkpoints/best.pt` and upload it.\n"
            "4. Come back here and try again.\n\n"
            f"<details><summary>Technical detail</summary>\n\n{MODEL_NOT_TRAINED_MSG}\n\n"
            f"{(_pipeline_error or '')}</details>"
        )
    try:
        result = predict_video(video_path, pipeline=pipe)
    except Exception:  # noqa: BLE001
        return "## ❌ Error during analysis\n\n```\n" + traceback.format_exc(limit=5) + "\n```"
    return _format_result(result)


demo = gr.Interface(
    fn=detect,
    inputs=gr.Video(label="Upload a video (MP4, short clip works best)"),
    outputs=gr.Markdown(label="Result"),
    title="🔍 Deepfake Video Detector",
    description=(
        "FYP — BSCS, IBADAT International University Islamabad (Ammad Ullah). "
        "Upload a video: the system extracts frames, detects faces with MTCNN, "
        "scores each face with a fine-tuned EfficientNet-B0 classifier, and "
        "aggregates the scores into a FAKE / REAL verdict. "
        "First run downloads the face-detector weights (~100 MB) and may take a few minutes on CPU."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
