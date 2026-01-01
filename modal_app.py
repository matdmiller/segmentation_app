"""Modal deployment for SAM 3 image segmentation.

This module defines a Modal app that loads Meta's SAM 3 model on an A10G GPU
and exposes a `segment` function for remote execution.
"""

from __future__ import annotations

import io
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import modal
from PIL import Image

app = modal.App("sam3-segmentation")

# Modal image with necessary dependencies for SAM 3 inference.
# Using A10G as requested to keep GPU costs constrained.
base_image = (
    modal.Image.debian_slim()
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        "numpy",
        "pillow",
        "opencv-python-headless",
        "git+https://github.com/facebookresearch/segment-anything-3.git",
    )
)


@app.cls(gpu="A10G", image=base_image)
class Sam3Worker:
    def __enter__(self):
        from sam3.modeling.sam3_image_encoder import SAM3ImageEncoder
        from sam3.modeling.sam3_image_model import SAM3ImageModel
        from sam3.sam3_image_predictor import SAM3ImagePredictor
        from sam3.utils.transforms import SAM3Transforms

        # Load default model weights shipped with the package.
        self.device = "cuda"
        self.model = SAM3ImageModel.from_pretrained("sam3_hiera_base_plus")
        self.model = self.model.to(self.device).eval()

        self.image_encoder = SAM3ImageEncoder.from_pretrained("sam3_hiera_base_plus")
        self.image_encoder = self.image_encoder.to(self.device).eval()

        self.predictor = SAM3ImagePredictor(
            image_encoder=self.image_encoder,
            image_model=self.model,
            transforms=SAM3Transforms(
                target_length=2560, pyramid_scales=(0.5, 0.75, 1.0, 1.25, 1.5)
            ),
        )
        return self

    def _load_image(self, data: bytes) -> Any:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img

    @modal.method()
    def segment(
        self,
        image_bytes: bytes,
        points: List[Tuple[float, float, str]],
        boxes: List[Tuple[float, float, float, float, str]],
        text_prompts: List[str],
    ) -> Dict[str, Any]:
        """Run SAM 3 inference with provided prompts.

        Args:
            image_bytes: Raw uploaded image bytes.
            points: List of (x, y, label) where label is "positive" or "negative".
            boxes: List of (x1, y1, x2, y2, label) bounding boxes with label similarly signed.
            text_prompts: Additional text prompts for SAM 3.
        """

        import numpy as np
        import torch

        image = self._load_image(image_bytes)
        width, height = image.size

        prompt_points = [((float(x), float(y)), 1 if label == "positive" else 0) for x, y, label in points]
        prompt_boxes = [
            ((float(x1), float(y1)), (float(x2), float(y2)), 1 if label == "positive" else 0)
            for x1, y1, x2, y2, label in boxes
        ]

        self.predictor.set_image(np.array(image), image_format="RGB")
        masks = []

        # Combine prompts for a single forward pass.
        pred = self.predictor.predict(
            point_inputs=[p for p, _ in prompt_points] if prompt_points else None,
            point_labels=[lbl for _, lbl in prompt_points] if prompt_points else None,
            box_inputs=[(b1, b2) for b1, b2, _ in prompt_boxes] if prompt_boxes else None,
            box_labels=[lbl for _, _, lbl in prompt_boxes] if prompt_boxes else None,
            text_prompts=text_prompts if text_prompts else None,
            multimask_output=True,
        )

        for mask_id, mask in enumerate(pred[0]):
            mask_binary = mask.astype(np.uint8)
            area = int(mask_binary.sum())
            if area == 0:
                continue
            coords = np.column_stack(np.nonzero(mask_binary))
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            masks.append(
                {
                    "id": f"mask-{uuid.uuid4().hex[:8]}",
                    "area": area,
                    "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
                    "points": coords.tolist(),
                }
            )

        return {
            "width": width,
            "height": height,
            "masks": masks,
            "prompt_points": points,
            "prompt_boxes": boxes,
            "text_prompts": text_prompts,
        }


@app.local_entrypoint()
def main(image_path: str):
    """Helper for local testing via `modal run modal_app.py --image-path path`."""
    payload = Path(image_path).read_bytes()
    worker = Sam3Worker()
    with worker:
        result = worker.segment.remote(payload, [], [], [])
    print(json.dumps(result, indent=2))
