"""Deepfake video detector demo — pretrained classifier, no training needed.

Pipeline: video -> frames (2 fps, max 60) -> MTCNN largest-face crop ->
pretrained SigLIP2 deepfake classifier (king1oo1/deepfake-model, Apache 2.0) ->
per-frame P(FAKE) -> temporal aggregation -> FAKE / REAL verdict JSON.

Scorers (see docs/EVALUATION.md for how these numbers were measured):
  --scorer classifier  (DEFAULT) the raw pretrained classification head,
                       threshold 0.5. The only head verified end-to-end on
                       video: it classifies both bundled sample videos
                       correctly (EVALUATION.md section 7).
  --scorer probe       logistic-regression head on frozen SigLIP2
                       embeddings, fitted ONLY on the tuning split of the
                       evaluation dataset; held-out test accuracy 96.8% on
                       that still-image dataset. Opt-in for video: the
                       probes were fitted on stills and misclassify the
                       bundled fake sample video (section 7).
                       Alternative head: --probe-file models/probe_celeb.npz
                       (fitted on Celeb-DF video frames; 87.4% held-out
                       on stills).

Usage:
    python run_demo.py --video data/test_real.mp4
    python run_demo.py --video path/to/video.mp4 --model-dir /path/to/model

Runs on CPU. Prints a JSON result to stdout.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ----------------------------------------------------------------------------
# Paths (overridable via CLI)
# ----------------------------------------------------------------------------
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models" / "pretrained" / "siglip2-deepfake"

MODEL_MISSING_MSG = (
    "Pretrained model not found at '{path}'. Expected files: config.json, "
    "preprocessor_config.json, model.safetensors. Download with:\n"
    "  python3 -c \"from huggingface_hub import snapshot_download; "
    "snapshot_download('king1oo1/deepfake-model', local_dir='{path}')\""
)

# Label mapping from the model's config.json: 0 = Real, 1 = Fake.
FAKE_ID = 1

# ----------------------------------------------------------------------------
# Fitted probe head (docs/EVALUATION.md). A logistic regression on frozen
# SigLIP2 vision embeddings, fitted ONLY on the tuning split of the
# evaluation dataset (220 images); the threshold stored in the file was also
# chosen on the tuning split. Held-out test accuracy: 96.8% (220 images).
# ----------------------------------------------------------------------------
PROBE_PATH = Path(__file__).resolve().parent / "models" / "probe.npz"
CLASSIFIER_THRESHOLD = 0.5


def load_probe(path=PROBE_PATH):
    """Load the fitted probe head, or return None if the file is absent."""
    path = Path(path)
    if not path.is_file():
        return None
    z = np.load(path)
    return {
        "coef": z["coef"].reshape(-1).astype(np.float64),
        "intercept": float(z["intercept"].reshape(-1)[0]),
        "mean": z["mean"].astype(np.float64),
        "scale": z["scale"].astype(np.float64),
        "threshold": float(z["threshold"]),
    }


def probe_prob(embedding, probe):
    """P(FAKE) from a SigLIP2 pooler embedding via the fitted probe head."""
    x = (np.asarray(embedding, dtype=np.float64) - probe["mean"]) / probe["scale"]
    logit = float(x @ probe["coef"] + probe["intercept"])
    return 1.0 / (1.0 + np.exp(-logit))


def load_classifier(model_dir):
    """Load SigLIP2 image processor + classification model from local dir."""
    model_dir = Path(model_dir)
    needed = ["config.json", "preprocessor_config.json", "model.safetensors"]
    missing = [f for f in needed if not (model_dir / f).is_file()]
    if missing:
        raise FileNotFoundError(
            MODEL_MISSING_MSG.format(path=model_dir)
            + f"\nMissing: {missing}"
        )
    # NOTE: the model card was authored with transformers 5.x ("SiglipImageProcessorFast"),
    # which does not exist in the 4.x line. We construct the equivalent slow
    # processor explicitly from the values in preprocessor_config.json:
    # 224x224, rescale 1/255, normalize mean=0.5/std=0.5.
    from transformers import SiglipImageProcessor, AutoModelForImageClassification

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
        str(model_dir), trust_remote_code=False, local_files_only=True
    )
    model.eval()
    return processor, model


def load_mtcnn(device):
    """MTCNN face detector. We use detect() + manual PIL cropping (instead of
    the MTCNN __call__ crop tensor) because the default post_process
    standardizes pixel values, which corrupts the classifier input."""
    from facenet_pytorch import MTCNN

    return MTCNN(keep_all=True, device=device, min_face_size=40)


def largest_face_crop(pil_img, mtcnn, margin=20):
    """Return the largest detected face as a PIL crop, or None."""
    boxes, _ = mtcnn.detect(pil_img)
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
    x1 = max(0, int(x1 - margin))
    y1 = max(0, int(y1 - margin))
    x2 = min(pil_img.width, int(x2 + margin))
    y2 = min(pil_img.height, int(y2 + margin))
    if x2 <= x1 or y2 <= y1:
        return None
    return pil_img.crop((x1, y1, x2, y2))


def extract_frames(video_path, fps=2, max_frames=60):
    """Sample frames with cv2; returns list of (PIL image, timestamp_sec)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(src_fps / fps))
    frames = []
    idx = 0
    while len(frames) < max_frames:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((Image.fromarray(rgb), idx / src_fps))
        idx += 1
    cap.release()
    return frames


def score_faces(frames, mtcnn, processor, model, device, probe=None):
    """Detect largest face per frame and return [(timestamp, p_fake)].

    With `probe`, p_fake comes from the fitted logistic-regression head on
    the frozen SigLIP2 embedding of the face crop; otherwise from the raw
    pretrained classification head.
    """
    scored = []
    with torch.no_grad():
        for pil_img, ts in frames:
            face_pil = largest_face_crop(pil_img, mtcnn)
            if face_pil is None:
                continue  # no face: carries no signal, skip quietly
            inputs = processor(images=face_pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            if probe is not None:
                emb = model.vision_model(
                    pixel_values=inputs["pixel_values"]).pooler_output
                p_fake = probe_prob(emb.cpu().numpy()[0], probe)
            else:
                logits = model(**inputs).logits
                p_fake = torch.softmax(logits, dim=-1)[0, FAKE_ID].item()
            scored.append((ts, p_fake))
    return scored


def aggregate(scored, fake_threshold=0.5, suspicious_threshold=0.7):
    """Temporal aggregation -> verdict dict (JSON-serialisable)."""
    if not scored:
        return {
            "verdict": "NO_FACE_DETECTED",
            "confidence": 0.0,
            "video_score": 0.0,
            "frames_analyzed": 0,
            "suspicious_segments": [],
            "frame_scores": [],
            "frame_timestamps": [],
            "note": "No faces detected in sampled frames; try a video with a visible front-facing person.",
        }
    scores = np.array([s for _, s in scored])
    ts = np.array([t for t, _ in scored])
    # Focus on the most suspicious half: fake artefacts appear only in some frames.
    k = max(1, len(scores) // 2)
    video_score = float(np.mean(np.sort(scores)[-k:]))
    verdict = "FAKE" if video_score >= fake_threshold else "REAL"
    confidence = video_score if verdict == "FAKE" else 1.0 - video_score

    # Suspicious segments: contiguous runs of frames with score > 0.7.
    segments = []
    start = None
    peak = 0.0
    for t, s in scored:
        if s > suspicious_threshold:
            if start is None:
                start, peak = t, s
            else:
                peak = max(peak, s)
        else:
            if start is not None:
                segments.append(
                    {"start_sec": round(start, 2), "end_sec": round(t, 2),
                     "peak_score": round(peak, 3)}
                )
                start = None
    if start is not None:
        segments.append(
            {"start_sec": round(start, 2), "end_sec": round(ts[-1], 2),
             "peak_score": round(peak, 3)}
        )

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "video_score": round(video_score, 4),
        "frames_analyzed": len(scored),
        "suspicious_segments": segments,
        "frame_scores": [round(float(s), 3) for s in scores],
        "frame_timestamps": [round(float(t), 2) for t in ts],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Deepfake video detector (pretrained SigLIP2, no training needed)."
    )
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR),
                        help="local dir with the pretrained model files")
    parser.add_argument("--fps", type=int, default=2, help="frames sampled per second")
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--scorer", choices=["probe", "classifier"],
                        default=None,
                        help="scoring head: raw classifier (default — the "
                             "only head verified end-to-end on video) or the "
                             "fitted probe (opt-in; best on still images, "
                             "see docs/EVALUATION.md sections 3 and 7)")
    parser.add_argument("--probe-file", default=str(PROBE_PATH),
                        help="which fitted probe head to use "
                             "(e.g. models/probe_celeb.npz, fitted on "
                             "Celeb-DF video frames — see docs/EVALUATION.md)")
    args = parser.parse_args(argv)

    device = torch.device("cpu")
    try:
        processor, model = load_classifier(args.model_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    model.to(device)
    mtcnn = load_mtcnn(device)

    probe = load_probe(args.probe_file) if args.scorer == "probe" else None
    if args.scorer == "probe" and probe is None:
        print(f"ERROR: --scorer probe requested but {args.probe_file} is missing.",
              file=sys.stderr)
        return 1
    scorer = "probe" if probe is not None else "classifier"
    threshold = probe["threshold"] if probe is not None else CLASSIFIER_THRESHOLD

    frames = extract_frames(args.video, fps=args.fps, max_frames=args.max_frames)
    if not frames:
        print("ERROR: no frames could be read from the video.", file=sys.stderr)
        return 1
    scored = score_faces(frames, mtcnn, processor, model, device, probe=probe)
    result = aggregate(scored, fake_threshold=threshold)
    result["scorer"] = scorer
    result["verdict_threshold"] = round(threshold, 4)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
