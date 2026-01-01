"""FastHTML-powered frontend for SAM 3 image segmentation via Modal.

Run locally with `python app.py` (serves on localhost:8000).
Ensure Modal credentials are configured in the environment.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

from fasthtml.common import (
    Button,
    Div,
    Form,
    H2,
    Input,
    Label,
    Link,
    Option,
    Script,
    Select,
    Style,
    Textarea,
    fast_app,
    serve,
)
from monsterui.all import Theme
from starlette.requests import Request
from starlette.responses import RedirectResponse, Response
from PIL import Image, ImageDraw, ImageFile

modal = None
if importlib.util.find_spec("modal"):
    import modal  # type: ignore

DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

ImageFile.LOAD_TRUNCATED_IMAGES = True

app, rt = fast_app(hdrs=Theme.blue.headers(mode="dark", daisy=True), live=False)


class SessionState:
    """In-memory state container."""

    def __init__(self):
        self.image_path: Path | None = None
        self.predictions: Dict[str, object] = {}

    def reset(self):
        self.image_path = None
        self.predictions = {}


global_state = SessionState()


def load_modal_function():
    if modal is None:
        return None
    try:
        stub = modal.Stub.from_name("sam3-segmentation")
        worker = getattr(getattr(stub, "cls", None), "sam3worker", None)
        return worker.segment if worker else None
    except Exception:
        return None


@rt("/")
def home():
    return Div(
        Style(
            """
            body { margin: 0; background: #0b0c10; }
            canvas, img { max-width: 100%; border-radius: 12px; }
            .stack { position: relative; width: 100%; }
            .stack canvas, .stack img { position: absolute; top: 0; left: 0; }
            .stack img { position: relative; }
            """
        ),
        Div(
            H2("SAM 3 Image Segmentor"),
            Div(Link("Requirements", href="/requirements", target="_blank"), _class="badge badge-outline"),
            Form(
                Label("Upload an image", _class="label-text"),
                Input(type="file", name="image", accept="image/*", required=True, _class="file-input file-input-bordered w-full"),
                Button("Upload", type="submit", _class="btn btn-primary"),
                method="post",
                action="/upload",
                enctype="multipart/form-data",
            ),
            Div(_class="divider opacity-40"),
            Div(
                Div(
                    Label("Selection type", _class="label-text"),
                    Select(
                        Option("Point", value="point"),
                        Option("Box", value="box"),
                        Option("Text", value="text"),
                        name="mode",
                        id="mode",
                        _class="select select-bordered w-full",
                    ),
                ),
                Div(
                    Label("Polarity", _class="label-text"),
                    Select(
                        Option("Positive", value="positive"),
                        Option("Negative", value="negative"),
                        name="polarity",
                        id="polarity",
                        _class="select select-bordered w-full",
                    ),
                ),
                Div(
                    Label("Text prompt", _class="label-text"),
                    Input(name="text_prompt", id="text_prompt", placeholder="e.g., select the cat", _class="input input-bordered w-full"),
                ),
                Div(
                    Label("Selections JSON", _class="label-text"),
                    Textarea(id="selections", rows=4, readonly=True, _class="textarea textarea-bordered w-full font-mono"),
                ),
                Div(
                    Label("Predictions JSON", _class="label-text"),
                    Textarea(id="predictions", rows=8, readonly=True, _class="textarea textarea-bordered w-full font-mono"),
                ),
                Div(
                    Button("Run segmentation", id="run", _class="btn btn-accent"),
                    Button("Download predictions", id="download", _class="btn"),
                    _class="flex gap-3",
                ),
                _class="flex flex-col gap-4",
                id="controls",
            ),
            Div(_class="divider opacity-40"),
            Div(id="preview"),
            _class="card bg-base-200 shadow-2xl max-w-5xl mx-auto p-6 space-y-4",
        ),
        Script(
            """
            const state = { selections: [], imageUrl: null, imageSize: null, overlay: null };
            const preview = document.getElementById('preview');
            const selectionsBox = document.getElementById('selections');
            const predictionsBox = document.getElementById('predictions');
            const textPrompt = document.getElementById('text_prompt');
            const mode = document.getElementById('mode');
            const polarity = document.getElementById('polarity');

            function renderPreview() {
                preview.innerHTML = '';
                if (!state.imageUrl) { return; }
                const container = document.createElement('div');
                container.className = 'stack';
                const img = document.createElement('img');
                img.src = state.imageUrl;
                img.onload = () => {
                    state.imageSize = { width: img.width, height: img.height };
                    const canvas = document.createElement('canvas');
                    canvas.width = img.width;
                    canvas.height = img.height;
                    canvas.id = 'overlay';
                    canvas.addEventListener('click', (ev) => handleCanvasClick(ev, canvas));
                    container.appendChild(img);
                    container.appendChild(canvas);
                    preview.appendChild(container);
                    redraw();
                };
            }

            function handleCanvasClick(ev, canvas) {
                const rect = canvas.getBoundingClientRect();
                const x = ev.clientX - rect.left;
                const y = ev.clientY - rect.top;
                const entry = { id: crypto.randomUUID(), mode: mode.value, polarity: polarity.value, coords: [x, y] };
                if (mode.value === 'box') {
                    if (!state.partialBox) {
                        state.partialBox = { x1: x, y1: y };
                        return;
                    } else {
                        entry.coords = [state.partialBox.x1, state.partialBox.y1, x, y];
                        state.partialBox = null;
                    }
                }
                if (mode.value === 'text') {
                    entry.text = textPrompt.value;
                }
                state.selections.push(entry);
                selectionsBox.value = JSON.stringify(state.selections, null, 2);
                redraw();
            }

            function redraw() {
                const canvas = document.getElementById('overlay');
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.lineWidth = 2;
                state.selections.forEach(sel => {
                    ctx.strokeStyle = sel.polarity === 'positive' ? '#22c55e' : '#ef4444';
                    ctx.fillStyle = sel.polarity === 'positive' ? 'rgba(34,197,94,0.25)' : 'rgba(239,68,68,0.25)';
                    if (sel.mode === 'point') {
                        ctx.beginPath();
                        ctx.arc(sel.coords[0], sel.coords[1], 6, 0, 2 * Math.PI);
                        ctx.fill();
                    } else if (sel.mode === 'box') {
                        const [x1, y1, x2, y2] = sel.coords;
                        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
                    }
                });
                if (state.overlay) {
                    const maskImg = new Image();
                    maskImg.src = state.overlay;
                    maskImg.onload = () => {
                        ctx.drawImage(maskImg, 0, 0);
                    }
                }
            }

            async function runSegmentation() {
                const payload = { selections: state.selections, text_prompt: textPrompt.value };
                const res = await fetch('/segment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (!res.ok) { alert('Segmentation failed'); return; }
                const data = await res.json();
                state.overlay = data.overlay;
                predictionsBox.value = JSON.stringify(data.predictions, null, 2);
                redraw();
            }

            async function downloadPredictions() {
                const blob = new Blob([predictionsBox.value], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'predictions.json';
                a.click();
                URL.revokeObjectURL(url);
            }

            document.getElementById('run').addEventListener('click', (e) => { e.preventDefault(); runSegmentation(); });
            document.getElementById('download').addEventListener('click', (e) => { e.preventDefault(); downloadPredictions(); });

            fetch('/session').then(r => r.json()).then(data => {
                if (data.image_url) {
                    state.imageUrl = data.image_url;
                    renderPreview();
                }
                if (data.predictions) {
                    predictionsBox.value = JSON.stringify(data.predictions, null, 2);
                }
            });

            window.renderPreview = renderPreview;
            window.redraw = redraw;
            renderPreview();
            """
        ),
        _class="min-h-screen text-base-content p-6 flex justify-center",
    )


@rt("/requirements")
def show_requirements():
    return Link("Open requirements", href="/static/requirements", target="_blank")


@rt("/static/requirements")
def serve_requirements():
    return Path("requirements.md").read_text()


@rt("/session")
def session_state():
    image_url = None
    if global_state.image_path and global_state.image_path.exists():
        raw = global_state.image_path.read_bytes()
        image_url = "data:image/png;base64," + base64.b64encode(raw).decode()
    return {
        "image_url": image_url,
        "predictions": global_state.predictions or {},
    }


@rt("/image/{image_id:path}")
def image(image_id: str):
    path = DATA_DIR / image_id
    if not path.exists():
        return Response(content=b"", status_code=404)
    data = path.read_bytes()
    return Response(content=data, media_type="image/png")


@rt("/upload", methods=["POST"])
async def upload(req: Request):
    form = await req.form()
    file = form.get("image")
    if file is None:
        return {"error": "No file provided"}
    raw = await file.read()
    if not raw:
        return {"error": "Empty upload"}
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return {"error": "Invalid image"}
    fname = f"{uuid.uuid4().hex}.png"
    path = DATA_DIR / fname
    img.save(path)
    global_state.image_path = path
    global_state.predictions = {}
    return RedirectResponse(url="/", status_code=302)


def overlay_masks(base_image: Image.Image, masks: List[Dict[str, object]]) -> str:
    canvas = base_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(canvas, "RGBA")
    palette = [(34, 197, 94, 90), (6, 182, 212, 90), (239, 68, 68, 90), (234, 179, 8, 90)]
    for idx, mask in enumerate(masks):
        coords = mask.get("points", [])
        if not coords:
            continue
        xs = [p[1] for p in coords]
        ys = [p[0] for p in coords]
        points = list(zip(xs, ys))
        draw.polygon(points, fill=palette[idx % len(palette)])
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@rt("/segment", methods=["POST"])
def segment(payload: dict):
    if not global_state.image_path:
        return {"error": "No image uploaded"}
    selections = payload.get("selections", [])
    text_prompt = payload.get("text_prompt") or ""

    points: List[Tuple[float, float, str]] = []
    boxes: List[Tuple[float, float, float, float, str]] = []
    for sel in selections:
        mode = sel.get("mode")
        polarity = sel.get("polarity", "positive")
        coords = sel.get("coords", [])
        if mode == "point" and len(coords) >= 2:
            points.append((coords[0], coords[1], polarity))
        elif mode == "box" and len(coords) >= 4:
            boxes.append((coords[0], coords[1], coords[2], coords[3], polarity))
    text_prompts = [text_prompt] if text_prompt else []

    runner = load_modal_function()
    image_bytes = global_state.image_path.read_bytes()
    predictions = None
    if runner:
        predictions = runner.remote(image_bytes, points, boxes, text_prompts)
    else:
        # Local fallback for UI testing without Modal.
        predictions = {
            "width": 0,
            "height": 0,
            "masks": [],
            "prompt_points": points,
            "prompt_boxes": boxes,
            "text_prompts": text_prompts,
        }

    overlay = overlay_masks(Image.open(global_state.image_path), predictions.get("masks", []))
    global_state.predictions = predictions
    return {"overlay": overlay, "predictions": predictions}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    serve(port=port, host="0.0.0.0", reload=False)
