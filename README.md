# YOLO-World + SAM 2 — Zero-Shot Detection + Segmentation

> **Live Demo Website:** [https://remandey-segmentation.hf.space](https://remandey-segmentation.hf.space)
>
> **Demonstration:** [https://youtu.be/Y62N2Kmhm80](https://youtu.be/Y62N2Kmhm80)
>
> [![Watch the demonstration](https://img.youtube.com/vi/Y62N2Kmhm80/hqdefault.jpg)](https://youtu.be/Y62N2Kmhm80)

Detect **any object described in plain words** and segment every instance —
no training, no fixed class list, no GPU of your own required.

Upload an image → type a prompt (`blue A4 sheet box`) → set confidence →
**RUN SEGMENTATION**. YOLO-World finds all matching instances, SAM 2 cuts out
pixel-precise masks for each one.

- **Open-vocabulary:** arbitrary prompts, comma-separated multi-class
  (`red carton, blue box`)
- **Multi-instance:** all detections segmented, not just the top box
- **ZeroGPU-ready:** serverless A100 per request on Hugging Face's free tier,
  CPU fallback everywhere else
- **Single file app:** `app.py` + `requirements.txt`, nothing else to deploy

## How it works

```text
PIL image ──► resize (≤ MAX_SIDE) ──► BGR numpy
                                          │
              ┌───────────────────────────┘
              ▼
   YOLO-World (text prompt → classes)
              │  boxes + confidences + class ids (ALL boxes)
              ▼
   SAM 2 (bboxes=[...] → masks, one per box)
              │
              ▼
   supervision: green masks + boxes overlaid
              │
              ▼
   Segmented image │ Mask image │ Detection table │ Status line
```

Details worth knowing:

1. `detector.set_classes(classes)` re-grounds YOLO-World to your prompt on every
   request — the model itself is loaded once and reused.
2. All YOLO boxes are passed to SAM 2 in a single call (`bboxes=[...]`), so
   N detections yield N masks in one shot.
3. `sv.Detections.from_ultralytics(...)` converts SAM output for annotation;
   if SAM throws, the app falls back to **box-only output** instead of failing.
4. On ZeroGPU, CUDA exists only inside the `@spaces.GPU` call, so models are
   (re)moved to `cuda` per request with explicit `device=` on every predict;
   off HF the same code runs on CPU untouched.

## Project structure

```text
.
├── app.py            # the whole app: models, pipeline, Gradio UI
├── requirements.txt  # pinned deps incl. `spaces` (ZeroGPU) + gradio
├── README.md         # this file (frontmatter = Space config)
├── .gitignore        # ignores *.pt, weights/, runs/, outputs
├── main.py           # original local reference script (single-box version)
├── test.py           # scratch file, not used by the app
└── test1.png         # sample input image
```

`weights/`, `*.pt`, `*_output.jpg` are local artifacts — git-ignored, never
needed for deployment since weights auto-download (see below).

## Requirements

| Package | Version | Why |
|---|---|---|
| `gradio` | 5.6.0 | UI + Space runtime |
| `spaces` | ≥ 0.28.0 | ZeroGPU `@spaces.GPU` decorator (no-op locally) |
| `ultralytics` | 8.3.40 | `YOLOWorld` + `SAM` |
| `supervision` | 0.25.1 | mask/box annotators |
| `opencv-python-headless` | 4.10.0.84 | image convert/resize/overlay |
| `numpy` | 1.26.4 | array plumbing |
| `Pillow` | 10.4.0 | Gradio image I/O |
| `torch` / `torchvision` | 2.4.1 / 0.19.1 | backend for both models |

Python 3.10+ recommended.

## Models

Weights resolve **automatically**: `YOLOWorld("...pt")` / `SAM("...pt")` download
from Ultralytics release assets into the cache on first boot. First startup is
slow (download + warmup); everything after is fast. No manual upload, no extra
Space configuration.

| Role | Default (CPU/ZeroGPU-friendly) | Larger alternative |
|---|---|---|
| Detection | `yolov8s-worldv2.pt` (~30 MB) | `yolov8m-worldv2.pt`, `yolov8l-worldv2.pt`, `yolov8x-worldv2.pt` |
| Segmentation | `sam2_t.pt` (tiny) | `sam2_s.pt`, `sam2_b.pt`, `sam2_l.pt` |

Bigger = more accurate but slower and hungrier (quota/RAM). The `x` + `b` combo
wants a real GPU; the defaults run anywhere.

## Configuration (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `YOLO_WEIGHTS` | `yolov8s-worldv2.pt` | detection checkpoint |
| `SAM_WEIGHTS` | `sam2_t.pt` | segmentation checkpoint |
| `MAX_SIDE` | `1024` | images downscaled so longest side ≤ this (speed/RAM) |

```bash
YOLO_WEIGHTS=yolov8x-worldv2.pt SAM_WEIGHTS=sam2_b.pt MAX_SIDE=2048 python app.py
```

The GPU time limit lives in code: `@spaces.GPU(duration=120)` in `app.py`
(max 120 s of A100 per click — raise it if you use large models).

## Run locally

```bash
pip install -r requirements.txt
python app.py
# open the printed http://127.0.0.1:7860 URL
```

The `spaces` decorator does nothing off Hugging Face, so local runs just use
your CPU (or CUDA if `torch.cuda.is_available()`).

## Deploy on Hugging Face Spaces (ZeroGPU)

ZeroGPU = free serverless A100s billed per GPU-second with a pooled quota
(tighter on the free tier). Idle costs nothing; GPU attaches only while
`segment()` runs.

1. Create a Space: https://huggingface.co/new-space → SDK **Gradio**
   (this repo's frontmatter already declares `sdk`, `sdk_version`, `app_file`).
2. Push these three files to the Space root:
   ```bash
   git clone https://huggingface.co/spaces/<USER>/<SPACE>
   cp app.py requirements.txt README.md <SPACE>/
   cd <SPACE> && git add . && git commit -m "Zero-shot YOLO-World + SAM2" && git push
   ```
   (`.pt` files must NOT be pushed — they're git-ignored and auto-download.)
3. Space **Settings → Hardware → ZeroGPU**. The build restarts; watch **Logs**
   for `[boot] models loaded`.
4. Open the Space, upload an image, run. Check the status line confirms `cuda`.

No `Dockerfile`, no secrets, no extra setup. To go back to plain CPU hosting,
switch hardware to CPU — no code change needed.

## Usage guide

**Inputs**

- *Image* — PNG/JPG/JPEG/WEBP, anything reasonable; auto-downscaled to
  `MAX_SIDE`, so huge photos don't blow RAM/quota.
- *Prompt* — 1–10 classes, comma-separated. Good: `blue A4 sheet box`,
  `foam roller`, `red carton, blue box`. Bad: whole sentences, >200 chars
  (rejected with a message).
- *Confidence* — 0.01–1.0, default 0.15. Lower = more boxes (more recall, more
  noise); raise toward 0.3–0.5 if you get false positives.

**Outputs**

- *Segmented* — green masks + boxes over the original.
- *Mask* — green-on-black mask canvas (one channel view of all masks).
- *Detections* — `# │ Class │ Conf │ Box [x1,y1,x2,y2]` (pixel coords, one row
  per instance).
- *Status* — `Found N object(s) in Ts on cuda/cpu`, or guidance when empty.

## API access

Gradio exposes the function as a REST endpoint — point any client at it:

```bash
curl -X POST https://<USER>-<SPACE>.hf.space/gradio_api/call/segment \
  -H "Content-Type: application/json" \
  -d '{"data": [{"path": "test1.png"}, "blue A4 sheet box", 0.15]}'
```

(Interactive docs: `https://<USER>-<SPACE>.hf.space/?view=api`. Image input
also accepts a URL string in place of the upload dict.)

## Performance expectations

| Setup | Per-run (1024px, default models) |
|---|---|
| ZeroGPU (A100, attached) | seconds |
| Local CPU | tens of seconds – ~2 min (SAM 2 dominates) |
| HF free CPU hardware | same as local CPU, slower under load |

Keep runs cheap: small images, tiny models, one request at a time. Large
weights + large images multiply ZeroGPU quota burn and risk hitting `duration`.

## Troubleshooting

| Symptom | Likely cause → fix |
|---|---|
| `Please upload an image` / `enter a prompt` | validation — fill the field |
| `No objects detected` | prompt too specific or conf too high → rephrase, lower to ~0.05–0.1 |
| First run very slow | weight download + cold start — one-time, retry after Logs settle |
| ZeroGPU quota error | free-tier minutes spent → wait for reset, shrink image/models |
| `duration` exceeded | large model/image blew 120 s → raise `@spaces.GPU(duration=…)` |
| Box-only output, no masks | SAM threw (logged `[warn]`) — still returns boxes; retry or smaller image |
| Build fails on Space | usually a `requirements.txt` drift → check Logs, keep pins as shipped |

## Limitations

- RGB only; heavy occlusion, tiny objects, and mirror/reflective surfaces
  degrade masks (same as the underlying models).
- Boxes are axis-aligned; grasp/contact reasoning is out of scope.
- ZeroGPU free quota is shared and throttled — this is a demo setup, not a
  production SLA.

## Notes

- `main.py` is the original single-box (`boxes[0]`) script, kept for reference;
  `app.py` segments **all** boxes.
- Never commit `*.pt`/`weights/` — if they were committed before the
  `.gitignore`, run `git rm --cached sam2_b.pt yolov8x-worldv2.pt weights/*`.
