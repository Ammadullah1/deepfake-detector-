"""Deepfake Video Detector — website demo for Hugging Face Spaces.

No terminal, no training, no manual model download: upload this project folder
to a Hugging Face Space (Gradio SDK) via the website, rename THIS file to
``app.py`` on the Space, and you get a public link that detects deepfakes in
uploaded videos.

Pipeline (the same tested pipeline as pretrained_demo.py):
video -> frames (2 fps) -> MTCNN largest-face crop -> pretrained SigLIP2
classifier (king1oo1/deepfake-model, Apache 2.0, downloaded automatically from
the Hugging Face Hub on first run) -> per-frame P(FAKE) -> temporal
aggregation -> FAKE / REAL verdict with confidence and suspicious segments.

Honest framing (see docs/PRETRAINED_MODEL.md): the classifier is a PUBLIC
pretrained model, not trained by us. Our original contribution is the
video-level pipeline: frame sampling, face extraction and temporal
aggregation. No accuracy claims: the two bundled sample videos are smoke
tests, not a benchmark.
"""

import sys
import traceback
from pathlib import Path

import gradio as gr
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pretrained_demo import (  # noqa: E402
    aggregate,
    extract_frames,
    load_mtcnn,
    score_faces,
)

MODEL_ID = "king1oo1/deepfake-model"  # public, Apache 2.0
SPACE_MAX_FRAMES = 40  # bound inference time on free CPU hardware

_pipeline = None
_pipeline_error = None


def load_classifier_hub():
    """Load the SigLIP2 processor + classification model from the HF Hub.

    Same explicit-processor workaround as pretrained_demo.py: the model card
    was authored with transformers 5.x ("SiglipImageProcessorFast"), which
    does not exist in the pinned 4.46.3 line, so we construct the equivalent
    processor explicitly (224x224, rescale 1/255, normalize mean=0.5/std=0.5).
    """
    from transformers import AutoModelForImageClassification, SiglipImageProcessor

    processor = SiglipImageProcessor(
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        size={"height": 224, "width": 224},
        rescale_factor=1.0 / 255.0,
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
    )
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID, trust_remote_code=False
    )
    model.eval()
    return processor, model


def get_pipeline():
    """Load everything lazily on first request (keeps Space startup fast)."""
    global _pipeline, _pipeline_error
    if _pipeline is None and _pipeline_error is None:
        try:
            device = torch.device("cpu")
            processor, model = load_classifier_hub()
            model.to(device)
            mtcnn = load_mtcnn(device)
            _pipeline = (processor, model, mtcnn, device)
        except Exception:  # noqa: BLE001 - show the real error in the UI
            _pipeline_error = traceback.format_exc(limit=5)
    return _pipeline


def _format_result(result: dict) -> str:
    verdict = result.get("verdict", "UNKNOWN")
    if verdict == "NO_FACE_DETECTED":
        return (
            "## ⚪ No face detected\n\n"
            f"{result.get('note', '')}\n\n"
            "Try a video with a visible, front-facing person."
        )
    conf = result.get("confidence", 0.0) * 100
    badge = (
        "🔴 **FAKE** — this video looks manipulated"
        if verdict == "FAKE"
        else "🟢 **REAL** — no manipulation detected"
    )
    bars = "█" * int(conf // 5) + "░" * (20 - int(conf // 5))
    md = f"## {badge}\n\n**Confidence:** {conf:.1f}%\n`{bars}`\n\n"
    md += f"Frames analyzed: **{result.get('frames_analyzed', 0)}**\n\n"
    segments = result.get("suspicious_segments", []) or []
    if segments:
        md += "### ⚠️ Suspicious segments\n"
        for seg in segments:
            md += (
                f"- {seg['start_sec']:.1f}s → {seg['end_sec']:.1f}s "
                f"(peak score {seg.get('peak_score', 0):.2f})\n"
            )
    else:
        md += "No suspicious segments found.\n"
    return md


def detect(video_path):
    if not video_path:
        return "Please upload a video first."
    pipe = get_pipeline()
    if pipe is None:
        return (
            "## ❌ Could not load the model\n\n"
            "The Space failed to download the public classifier on first run "
            "(it needs ~370 MB from the Hugging Face Hub plus face-detector "
            "weights). Check your connection and retry — the download is cached "
            "after the first success.\n\n"
            f"<details><summary>Technical detail</summary>\n\n```\n{_pipeline_error or ''}\n```\n</details>"
        )
    processor, model, mtcnn, device = pipe
    try:
        frames = extract_frames(video_path, fps=2, max_frames=SPACE_MAX_FRAMES)
        if not frames:
            return "## ❌ Could not read that video\n\nTry an MP4 file."
        scored = score_faces(frames, mtcnn, processor, model, device)
        result = aggregate(scored)
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
        "Upload a video: frames are sampled, faces detected with MTCNN, each face "
        "scored by a public pretrained SigLIP2 deepfake classifier "
        "(king1oo1/deepfake-model, Apache 2.0), and the scores aggregated into a "
        "FAKE / REAL verdict. First run downloads the model (~370 MB) and may "
        "take a few minutes on free CPU hardware.\n\n"
        "**Honest note for the viva:** the classifier is a public pretrained model "
        "(not trained by us); our original work is the video pipeline — frame "
        "sampling, face extraction and temporal aggregation. The two example videos "
        "are smoke tests, not a benchmark, so no accuracy % is claimed."
    ),
    examples=[
        [str(ROOT / "samples" / "test_real.mp4")],
        [str(ROOT / "samples" / "test_fake.mp4")],
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
