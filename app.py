"""FastHTML app for SAM 3 image segmentation via Modal."""

import base64
import io
import json
import logging
import os
import uuid
from pathlib import Path

import modal
import numpy as np
from fasthtml.common import *
from PIL import Image, ImageDraw
from starlette.responses import Response

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("segmentation_app")

app, rt = fast_app(pico=False, hdrs=[
    Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.12.24/dist/full.min.css"),
    Script(src="https://cdn.tailwindcss.com"),
])

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Session state - includes persistent selections
state = {
    "image_path": None,
    "predictions": None,
    "overlay_path": None,
    "points": [],  # [(x, y, polarity), ...]
    "boxes": [],   # [(x1, y1, x2, y2, polarity), ...]
    "text_prompt": "",
}

MASK_COLORS = [
    (59, 130, 246, 120),   # blue
    (34, 197, 94, 120),    # green
    (249, 115, 22, 120),   # orange
    (168, 85, 247, 120),   # purple
    (236, 72, 153, 120),   # pink
]


def get_modal_segment():
    app_name = os.environ.get("MODAL_APP_NAME", "sam3-segmentation")
    logger.info("Looking up Modal app: %s", app_name)
    cls = modal.Cls.from_name(app_name, "Sam3Worker")
    return cls().segment


def decode_rle(rle: dict) -> np.ndarray:
    """Decode RLE-encoded mask to binary array."""
    shape = rle["shape"]
    starts = rle["starts"]
    lengths = rle["lengths"]

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1

    return mask.reshape(shape)


def render_overlay(image_path: Path, predictions: dict) -> Path:
    """Render actual segmentation masks as overlays."""
    img = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))

    masks = predictions.get("masks", [])
    for i, mask_data in enumerate(masks):
        color = MASK_COLORS[i % len(MASK_COLORS)]

        # Try to decode RLE mask if available
        rle = mask_data.get("rle")
        if rle:
            try:
                mask_binary = decode_rle(rle)
                # Create colored mask
                mask_rgba = np.zeros((*mask_binary.shape, 4), dtype=np.uint8)
                mask_rgba[mask_binary == 1] = color
                mask_img = Image.fromarray(mask_rgba, mode="RGBA")

                # Resize if needed to match image size
                if mask_img.size != img.size:
                    mask_img = mask_img.resize(img.size, Image.NEAREST)

                overlay = Image.alpha_composite(overlay, mask_img)
            except Exception as e:
                logger.warning(f"Failed to decode mask RLE: {e}")
                # Fall back to bbox
                _draw_bbox(overlay, mask_data, color)
        else:
            # Fall back to bbox if no RLE
            _draw_bbox(overlay, mask_data, color)

        # Draw score label on bbox
        bbox = mask_data.get("bbox", [])
        if len(bbox) == 4:
            draw = ImageDraw.Draw(overlay)
            score = mask_data.get("score", 0)
            x1, y1 = bbox[0], bbox[1]
            draw.text((x1 + 4, y1 + 4), f"{score:.0%}", fill=(255, 255, 255, 255))
            # Draw bbox outline
            draw.rectangle(bbox, outline=color[:3] + (255,), width=2)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    out_path = DATA_DIR / f"overlay_{uuid.uuid4().hex}.png"
    result.save(out_path)
    return out_path


def _draw_bbox(overlay: Image.Image, mask_data: dict, color: tuple):
    """Draw bounding box fallback."""
    bbox = mask_data.get("bbox", [])
    if len(bbox) == 4:
        draw = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=color[:3] + (255,), width=2)


def img_to_url(path: Path) -> str:
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"


@rt("/")
def home():
    has_image = state["image_path"] and state["image_path"].exists()

    return Title("SAM 3 Segmentor"), Html(
        Head(
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.12.24/dist/full.min.css"),
            Script(src="https://cdn.tailwindcss.com"),
        ),
        Body(
            Div(
                # Header
                Div(
                    H1("SAM 3 Image Segmentor", cls="text-2xl font-bold"),
                    Form(
                        Input(type="file", name="image", accept="image/*", cls="file-input file-input-bordered file-input-sm"),
                        Button("Upload", type="submit", cls="btn btn-primary btn-sm"),
                        action="/upload", method="post", enctype="multipart/form-data",
                        cls="flex gap-2 items-center"
                    ),
                    cls="flex justify-between items-center mb-4"
                ),

                workspace() if has_image else Div(
                    P("Upload an image to begin", cls="text-gray-500 text-lg"),
                    cls="border-2 border-dashed border-gray-300 rounded-lg p-32 text-center"
                ),

                cls="p-4 max-w-7xl mx-auto", data_theme="light"
            )
        )
    )


def workspace():
    has_overlay = state["overlay_path"] and state["overlay_path"].exists()
    input_url = img_to_url(state["image_path"])
    overlay_url = img_to_url(state["overlay_path"]) if has_overlay else None

    # Pass current selections to JS (exclude RLE from JSON to avoid huge payloads)
    predictions_for_js = None
    if state["predictions"]:
        predictions_for_js = {
            **state["predictions"],
            "masks": [{k: v for k, v in m.items() if k != "rle"} for m in state["predictions"].get("masks", [])]
        }

    selections_json = json.dumps({
        "points": state["points"],
        "boxes": state["boxes"],
        "text_prompt": state["text_prompt"],
        "predictions": predictions_for_js,
    })

    return Div(
        # Toolbar
        Div(
            Div(
                Label("Mode:", cls="text-sm font-medium mr-1"),
                Select(
                    Option("Box", value="box", selected=True),
                    Option("Point", value="point"),
                    id="mode-select", cls="select select-sm select-bordered"
                ),
                cls="flex items-center"
            ),
            Div(
                Label("Type:", cls="text-sm font-medium mr-1"),
                Select(
                    Option("Positive", value="positive", selected=True),
                    Option("Negative", value="negative"),
                    id="polarity-select", cls="select select-sm select-bordered"
                ),
                cls="flex items-center"
            ),
            Input(type="text", id="text-prompt", placeholder="Text prompt...",
                  value=state["text_prompt"], cls="input input-sm input-bordered w-40"),
            Button("Run", id="run-btn", cls="btn btn-sm btn-primary"),
            Button("Clear All", id="clear-btn", cls="btn btn-sm btn-outline"),
            Button("Download JSON", id="download-btn", cls="btn btn-sm btn-ghost") if state["predictions"] else None,
            Span(id="status", cls="text-sm text-gray-500 ml-2"),
            cls="flex flex-wrap gap-2 items-center p-2 bg-gray-100 rounded mb-3"
        ),

        # Input image (full width)
        Div(
            Div(
                Img(src=input_url, id="input-img", cls="w-full rounded shadow"),
                Canvas(id="canvas", cls="absolute top-0 left-0 w-full h-full"),
                cls="relative"
            ),
            P("Input — drag for boxes, click for points. Keys: b=box, p=point, t=positive, f=negative. Delete removes selection.", cls="text-xs text-gray-500 mt-1"),
            cls="mb-4"
        ),

        # Output image (full width)
        Div(
            Img(src=overlay_url or input_url, id="output-img",
                cls="w-full rounded shadow" + ("" if overlay_url else " opacity-30")),
            P("Output — Cmd/Ctrl+click to add positive bbox, Shift+Cmd/Ctrl+click for negative bbox" if overlay_url else "Run segmentation to see results",
              cls="text-xs text-gray-500 mt-1"),
            cls="mb-4"
        ),

        # Predictions JSON (collapsible)
        Details(
            Summary("Predictions JSON", cls="cursor-pointer text-sm font-medium text-gray-600"),
            Pre(json.dumps(predictions_for_js, indent=2) if predictions_for_js else "None",
                cls="text-xs bg-gray-100 p-2 rounded mt-2 max-h-64 overflow-auto"),
            cls="mt-2"
        ) if state["predictions"] else None,

        # Hidden data for JS and tests
        Script(f"window.SELECTIONS = {selections_json};"),
        Input(type="hidden", id="selections", value=json.dumps(state["points"] + state["boxes"])),
        Input(type="hidden", id="predictions", value=json.dumps(predictions_for_js) if predictions_for_js else ""),

        Script("""
(function() {
    const canvas = document.getElementById('canvas');
    const inputImg = document.getElementById('input-img');
    const outputImg = document.getElementById('output-img');
    const modeSelect = document.getElementById('mode-select');
    const polaritySelect = document.getElementById('polarity-select');
    const textInput = document.getElementById('text-prompt');
    const status = document.getElementById('status');

    if (!canvas || !inputImg) return;

    // Load persisted selections
    let points = (window.SELECTIONS && window.SELECTIONS.points) || [];
    let boxes = (window.SELECTIONS && window.SELECTIONS.boxes) || [];
    let predictions = (window.SELECTIONS && window.SELECTIONS.predictions) || null;
    if (textInput && window.SELECTIONS) textInput.value = window.SELECTIONS.text_prompt || '';

    // Interaction state
    let dragMode = null; // null, 'draw', 'move', 'resize-tl', 'resize-tr', 'resize-bl', 'resize-br', 'resize-t', 'resize-b', 'resize-l', 'resize-r'
    let startX = 0, startY = 0;
    let dragBoxIdx = -1;
    let dragStartBox = null;
    let selectedPoint = -1;
    let selectedBox = -1;

    const EDGE_THRESHOLD = 10; // pixels from edge to trigger resize

    function resize() {
        canvas.width = inputImg.clientWidth;
        canvas.height = inputImg.clientHeight;
        redraw();
    }

    inputImg.onload = resize;
    if (inputImg.complete) resize();
    window.addEventListener('resize', resize);

    function getScale() {
        return {
            x: inputImg.naturalWidth / canvas.width,
            y: inputImg.naturalHeight / canvas.height
        };
    }

    function toCanvas(px, py) {
        const s = getScale();
        return { x: px / s.x, y: py / s.y };
    }

    function toImage(cx, cy) {
        const s = getScale();
        return { x: cx * s.x, y: cy * s.y };
    }

    function findPoint(cx, cy) {
        for (let i = 0; i < points.length; i++) {
            const p = toCanvas(points[i][0], points[i][1]);
            const dist = Math.sqrt((cx - p.x) ** 2 + (cy - p.y) ** 2);
            if (dist < 12) return i;
        }
        return -1;
    }

    // Find box and determine if on edge (for resize) or inside (for move)
    function findBoxAndZone(cx, cy) {
        for (let i = boxes.length - 1; i >= 0; i--) {
            const [x1, y1, x2, y2] = boxes[i];
            const p1 = toCanvas(x1, y1);
            const p2 = toCanvas(x2, y2);

            // Check if within box bounds (with edge threshold)
            const inX = cx >= p1.x - EDGE_THRESHOLD && cx <= p2.x + EDGE_THRESHOLD;
            const inY = cy >= p1.y - EDGE_THRESHOLD && cy <= p2.y + EDGE_THRESHOLD;
            if (!inX || !inY) continue;

            // Determine zone
            const onLeft = Math.abs(cx - p1.x) < EDGE_THRESHOLD;
            const onRight = Math.abs(cx - p2.x) < EDGE_THRESHOLD;
            const onTop = Math.abs(cy - p1.y) < EDGE_THRESHOLD;
            const onBottom = Math.abs(cy - p2.y) < EDGE_THRESHOLD;

            if (onTop && onLeft) return { idx: i, zone: 'resize-tl' };
            if (onTop && onRight) return { idx: i, zone: 'resize-tr' };
            if (onBottom && onLeft) return { idx: i, zone: 'resize-bl' };
            if (onBottom && onRight) return { idx: i, zone: 'resize-br' };
            if (onTop) return { idx: i, zone: 'resize-t' };
            if (onBottom) return { idx: i, zone: 'resize-b' };
            if (onLeft) return { idx: i, zone: 'resize-l' };
            if (onRight) return { idx: i, zone: 'resize-r' };

            // Inside the box
            if (cx >= p1.x && cx <= p2.x && cy >= p1.y && cy <= p2.y) {
                return { idx: i, zone: 'move' };
            }
        }
        return { idx: -1, zone: null };
    }

    function updateCursor(cx, cy) {
        const { zone } = findBoxAndZone(cx, cy);
        if (zone === 'resize-tl' || zone === 'resize-br') canvas.style.cursor = 'nwse-resize';
        else if (zone === 'resize-tr' || zone === 'resize-bl') canvas.style.cursor = 'nesw-resize';
        else if (zone === 'resize-t' || zone === 'resize-b') canvas.style.cursor = 'ns-resize';
        else if (zone === 'resize-l' || zone === 'resize-r') canvas.style.cursor = 'ew-resize';
        else if (zone === 'move') canvas.style.cursor = 'move';
        else canvas.style.cursor = 'crosshair';
    }

    canvas.addEventListener('mousedown', function(e) {
        // Don't handle if modifier keys (let output image handle Cmd+click)
        if (e.metaKey || e.ctrlKey) return;

        const rect = canvas.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;
        if (cx < 0 || cy < 0 || cx > canvas.width || cy > canvas.height) return;

        startX = cx;
        startY = cy;

        // Check if clicking on a point first
        const pi = findPoint(cx, cy);
        if (pi >= 0) {
            if (selectedPoint === pi) {
                points.splice(pi, 1);
                selectedPoint = -1;
                status.textContent = 'Point deleted';
            } else {
                selectedPoint = pi;
                selectedBox = -1;
                status.textContent = 'Press Delete to remove point';
            }
            redraw();
            return;
        }

        // Check box interaction
        const { idx, zone } = findBoxAndZone(cx, cy);

        if (idx >= 0 && zone) {
            selectedBox = idx;
            selectedPoint = -1;
            dragBoxIdx = idx;
            dragStartBox = [...boxes[idx]];

            if (zone === 'move') {
                dragMode = 'move';
                status.textContent = 'Moving box...';
            } else {
                dragMode = zone;
                status.textContent = 'Resizing box...';
            }
            redraw();
            return;
        }

        // No existing element clicked - create new
        selectedPoint = -1;
        selectedBox = -1;

        const mode = modeSelect.value;
        const polarity = polaritySelect.value;

        if (mode === 'box') {
            dragMode = 'draw';
        } else {
            const img = toImage(cx, cy);
            points.push([img.x, img.y, polarity]);
            status.textContent = 'Point added (' + polarity + ')';
            redraw();
        }
    });

    canvas.addEventListener('mousemove', function(e) {
        const rect = canvas.getBoundingClientRect();
        const cx = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
        const cy = Math.max(0, Math.min(canvas.height, e.clientY - rect.top));

        if (!dragMode) {
            updateCursor(cx, cy);
            return;
        }

        if (dragMode === 'draw') {
            redrawWithPreview(cx, cy);
            return;
        }

        if (dragMode === 'move' && dragBoxIdx >= 0) {
            const dx = cx - startX;
            const dy = cy - startY;
            const s = getScale();
            const imgDx = dx * s.x;
            const imgDy = dy * s.y;

            boxes[dragBoxIdx][0] = dragStartBox[0] + imgDx;
            boxes[dragBoxIdx][1] = dragStartBox[1] + imgDy;
            boxes[dragBoxIdx][2] = dragStartBox[2] + imgDx;
            boxes[dragBoxIdx][3] = dragStartBox[3] + imgDy;
            redraw();
            return;
        }

        // Resize modes
        if (dragMode && dragMode.startsWith('resize-') && dragBoxIdx >= 0) {
            const img = toImage(cx, cy);
            const b = boxes[dragBoxIdx];
            const orig = dragStartBox;

            if (dragMode.includes('l')) b[0] = Math.min(img.x, orig[2] - 5);
            if (dragMode.includes('r')) b[2] = Math.max(img.x, orig[0] + 5);
            if (dragMode.includes('t')) b[1] = Math.min(img.y, orig[3] - 5);
            if (dragMode.includes('b')) b[3] = Math.max(img.y, orig[1] + 5);

            redraw();
        }
    });

    canvas.addEventListener('mouseup', function(e) {
        if (!dragMode) return;

        const rect = canvas.getBoundingClientRect();
        const cx = Math.max(0, Math.min(canvas.width, e.clientX - rect.left));
        const cy = Math.max(0, Math.min(canvas.height, e.clientY - rect.top));

        if (dragMode === 'draw') {
            if (Math.abs(cx - startX) > 5 && Math.abs(cy - startY) > 5) {
                const polarity = polaritySelect.value;
                const p1 = toImage(Math.min(startX, cx), Math.min(startY, cy));
                const p2 = toImage(Math.max(startX, cx), Math.max(startY, cy));
                boxes.push([p1.x, p1.y, p2.x, p2.y, polarity]);
                status.textContent = 'Box added (' + polarity + ')';
            }
        } else if (dragMode === 'move') {
            status.textContent = 'Box moved';
        } else if (dragMode && dragMode.startsWith('resize-')) {
            // Normalize box coordinates after resize
            const b = boxes[dragBoxIdx];
            const x1 = Math.min(b[0], b[2]);
            const y1 = Math.min(b[1], b[3]);
            const x2 = Math.max(b[0], b[2]);
            const y2 = Math.max(b[1], b[3]);
            boxes[dragBoxIdx] = [x1, y1, x2, y2, b[4]];
            status.textContent = 'Box resized';
        }

        dragMode = null;
        dragBoxIdx = -1;
        dragStartBox = null;
        redraw();
    });

    canvas.addEventListener('mouseleave', function() {
        if (dragMode === 'draw') {
            dragMode = null;
            redraw();
        }
    });

    // Cmd/Ctrl+click on OUTPUT image to add bbox from prediction
    // Shift+Cmd/Ctrl+click for negative bbox
    if (outputImg) {
        outputImg.addEventListener('click', function(e) {
            // Must have Cmd or Ctrl
            if (!e.metaKey && !e.ctrlKey) return;
            e.preventDefault();
            e.stopPropagation();

            if (!predictions || !predictions.masks || predictions.masks.length === 0) {
                status.textContent = 'No predictions to select from';
                return;
            }

            const rect = outputImg.getBoundingClientRect();
            const cx = e.clientX - rect.left;
            const cy = e.clientY - rect.top;
            const scaleX = outputImg.naturalWidth / rect.width;
            const scaleY = outputImg.naturalHeight / rect.height;
            const imgX = cx * scaleX;
            const imgY = cy * scaleY;

            // Polarity based on Shift key
            const polarity = e.shiftKey ? 'negative' : 'positive';

            // Find which mask bbox contains this point
            for (const mask of predictions.masks) {
                const bbox = mask.bbox || [];
                if (bbox.length !== 4) continue;
                const [x1, y1, x2, y2] = bbox;
                if (imgX >= x1 && imgX <= x2 && imgY >= y1 && imgY <= y2) {
                    boxes.push([x1, y1, x2, y2, polarity]);
                    status.textContent = 'Added prediction bbox (' + polarity + ')';
                    redraw();
                    return;
                }
            }

            status.textContent = 'Click was not inside any prediction bbox';
        });
        outputImg.style.cursor = 'pointer';
    }

    function syncSelectionsInput() {
        const selectionsInput = document.getElementById('selections');
        if (selectionsInput) {
            const allSelections = [
                ...points.map(p => ({ mode: 'point', x: p[0], y: p[1], polarity: p[2] })),
                ...boxes.map(b => ({ mode: 'box', x1: b[0], y1: b[1], x2: b[2], y2: b[3], polarity: b[4] }))
            ];
            selectionsInput.value = JSON.stringify(allSelections);
        }
    }

    function redraw() {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.setLineDash([]);
        syncSelectionsInput();

        // Draw boxes with resize handles
        boxes.forEach(function(b, i) {
            const p1 = toCanvas(b[0], b[1]);
            const p2 = toCanvas(b[2], b[3]);
            const isSelected = selectedBox === i;

            ctx.strokeStyle = b[4] === 'positive' ? '#22c55e' : '#ef4444';
            ctx.fillStyle = b[4] === 'positive' ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.1)';
            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.setLineDash(isSelected ? [5, 5] : []);
            ctx.fillRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
            ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);

            // Draw resize handles for selected box
            if (isSelected) {
                ctx.setLineDash([]);
                ctx.fillStyle = b[4] === 'positive' ? '#22c55e' : '#ef4444';
                const handleSize = 6;
                const handles = [
                    [p1.x, p1.y], [p2.x, p1.y], [p1.x, p2.y], [p2.x, p2.y], // corners
                    [(p1.x + p2.x) / 2, p1.y], [(p1.x + p2.x) / 2, p2.y], // top/bottom
                    [p1.x, (p1.y + p2.y) / 2], [p2.x, (p1.y + p2.y) / 2]  // left/right
                ];
                handles.forEach(([hx, hy]) => {
                    ctx.fillRect(hx - handleSize / 2, hy - handleSize / 2, handleSize, handleSize);
                });
            }
        });

        // Draw points
        points.forEach(function(p, i) {
            const cp = toCanvas(p[0], p[1]);
            ctx.fillStyle = p[2] === 'positive' ? '#22c55e' : '#ef4444';
            ctx.strokeStyle = selectedPoint === i ? '#000' : '#fff';
            ctx.lineWidth = selectedPoint === i ? 3 : 2;
            ctx.setLineDash([]);
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, selectedPoint === i ? 8 : 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
        });
    }

    function redrawWithPreview(cx, cy) {
        redraw();
        const ctx = canvas.getContext('2d');
        const polarity = polaritySelect.value;
        ctx.strokeStyle = polarity === 'positive' ? '#22c55e' : '#ef4444';
        ctx.fillStyle = polarity === 'positive' ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.fillRect(startX, startY, cx - startX, cy - startY);
        ctx.strokeRect(startX, startY, cx - startX, cy - startY);
    }

    document.getElementById('run-btn').onclick = async function() {
        const btn = this;
        btn.disabled = true;
        btn.textContent = 'Processing...';
        status.textContent = 'Sending to SAM 3...';

        try {
            const res = await fetch('/segment', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    points: points,
                    boxes: boxes,
                    text_prompt: textInput ? textInput.value : ''
                })
            });

            if (!res.ok) throw new Error(await res.text());

            const data = await res.json();
            predictions = data;
            status.textContent = 'Found ' + (data.masks ? data.masks.length : 0) + ' mask(s)';

            // Refresh page to show new overlay but keep selections
            window.location.reload();
        } catch (err) {
            status.textContent = 'Error: ' + err.message;
        }

        btn.disabled = false;
        btn.textContent = 'Run';
    };

    document.getElementById('clear-btn').onclick = async function() {
        points = [];
        boxes = [];
        selectedPoint = -1;
        selectedBox = -1;
        if (textInput) textInput.value = '';
        redraw();

        await fetch('/clear', { method: 'POST' });
        status.textContent = 'Cleared all';
        window.location.reload();
    };

    const downloadBtn = document.getElementById('download-btn');
    if (downloadBtn) {
        downloadBtn.onclick = async function() {
            const res = await fetch('/predictions');
            const data = await res.json();
            const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'predictions.json';
            a.click();
        };
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Skip if typing in an input field
        if (document.activeElement.tagName === 'INPUT') return;

        // Delete/Backspace - remove selected element
        if (e.key === 'Delete' || e.key === 'Backspace') {
            if (selectedPoint >= 0) {
                points.splice(selectedPoint, 1);
                selectedPoint = -1;
                status.textContent = 'Point deleted';
                redraw();
                e.preventDefault();
            } else if (selectedBox >= 0) {
                boxes.splice(selectedBox, 1);
                selectedBox = -1;
                status.textContent = 'Box deleted';
                redraw();
                e.preventDefault();
            }
        }
        // Escape - deselect
        if (e.key === 'Escape') {
            selectedPoint = -1;
            selectedBox = -1;
            redraw();
        }
        // Mode shortcuts: b=box, p=point
        if (e.key === 'b' || e.key === 'B') {
            modeSelect.value = 'box';
            status.textContent = 'Mode: Box';
            e.preventDefault();
        }
        if (e.key === 'p' || e.key === 'P') {
            modeSelect.value = 'point';
            status.textContent = 'Mode: Point';
            e.preventDefault();
        }
        // Polarity shortcuts: t=positive (true), f=negative (false)
        if (e.key === 't' || e.key === 'T') {
            polaritySelect.value = 'positive';
            status.textContent = 'Type: Positive';
            e.preventDefault();
        }
        if (e.key === 'f' || e.key === 'F') {
            polaritySelect.value = 'negative';
            status.textContent = 'Type: Negative';
            e.preventDefault();
        }
    });
})();
        """),
        id="workspace"
    )


@rt("/upload", methods=["POST"])
async def upload(request):
    form = await request.form()
    file = form.get("image")
    if not file:
        return RedirectResponse("/", status_code=302)

    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    path = DATA_DIR / f"{uuid.uuid4().hex}.png"
    img.save(path)

    state["image_path"] = path
    state["predictions"] = None
    state["overlay_path"] = None
    state["points"] = []
    state["boxes"] = []
    state["text_prompt"] = ""

    return RedirectResponse("/", status_code=302)


@rt("/segment", methods=["POST"])
async def segment(request):
    if not state["image_path"]:
        return Response("No image uploaded", status_code=400)

    body = await request.json()

    # Store selections in state for persistence
    state["points"] = body.get("points", [])
    state["boxes"] = body.get("boxes", [])
    state["text_prompt"] = body.get("text_prompt", "")

    # Convert to Modal format
    points = [(p[0], p[1], p[2]) for p in state["points"]]
    boxes = [(b[0], b[1], b[2], b[3], b[4]) for b in state["boxes"]]
    text_prompts = [state["text_prompt"]] if state["text_prompt"] else []

    logger.info("Segmenting: points=%d, boxes=%d, text=%s", len(points), len(boxes), text_prompts)
    for i, p in enumerate(points):
        logger.info("  Point %d: (%.1f, %.1f) polarity=%s", i, p[0], p[1], p[2])
    for i, b in enumerate(boxes):
        logger.info("  Box %d: (%.1f,%.1f)-(%.1f,%.1f) polarity=%s", i, b[0], b[1], b[2], b[3], b[4])

    image_bytes = state["image_path"].read_bytes()
    segment_fn = get_modal_segment()
    result = segment_fn.remote(image_bytes, points, boxes, text_prompts)

    logger.info("Modal result: %d masks", len(result.get("masks", [])))
    logger.info("  Result prompt_points: %s", result.get("prompt_points", []))
    logger.info("  Result prompt_boxes: %s", result.get("prompt_boxes", []))

    state["predictions"] = result

    if result.get("masks"):
        state["overlay_path"] = render_overlay(state["image_path"], result)
    else:
        state["overlay_path"] = None

    return result


@rt("/clear", methods=["POST"])
def clear():
    state["points"] = []
    state["boxes"] = []
    state["text_prompt"] = ""
    state["predictions"] = None
    state["overlay_path"] = None
    return {"status": "cleared"}


@rt("/predictions")
def predictions():
    """Return predictions without RLE data (consistent with UI display)."""
    if not state["predictions"]:
        return {}
    # Exclude RLE to keep JSON download size reasonable
    return {
        **state["predictions"],
        "masks": [{k: v for k, v in m.items() if k != "rle"} for m in state["predictions"].get("masks", [])]
    }


if __name__ == "__main__":
    serve(port=int(os.environ.get("PORT", 8000)))
