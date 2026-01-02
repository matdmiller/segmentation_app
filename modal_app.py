"""Modal deployment for SAM 3 image segmentation on A10G GPU."""

import base64
import io
import os
import uuid
import zlib
from pathlib import Path
from typing import Any

import modal

APP_NAME = os.environ.get("MODAL_APP_NAME", "sam3-segmentation")

app = modal.App(APP_NAME)


def download_model():
    """Download SAM3 model weights during image build (runs on GPU)."""
    import os
    import torch
    import sam3
    from sam3 import build_sam3_image_model

    sam3_dir = os.path.dirname(sam3.__file__)
    bpe_path = os.path.join(sam3_dir, "assets", "bpe_simple_vocab_16e6.txt.gz")

    print("Pre-downloading SAM3 model weights...")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # This triggers the HuggingFace download and caches the weights
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        enable_inst_interactivity=True,
        load_from_HF=True,
    )
    print("SAM3 model weights downloaded and cached")


# Build image with SAM 3 dependencies and baked-in weights
sam3_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy",
        "pillow",
        "opencv-python-headless",
        "huggingface_hub",
        "einops",
        "decord",
        "pycocotools",
        "psutil",
        "matplotlib",
        "scipy",
        "git+https://github.com/facebookresearch/sam3.git",
    )
    .run_function(download_model, gpu="A10G", secrets=[modal.Secret.from_name("huggingface-secret")])
)

# HF secret for gated models (must be consistent between local and remote)
HF_SECRET_NAME = "huggingface-secret"
secrets = [modal.Secret.from_name(HF_SECRET_NAME)]


def encode_mask_rle(mask_binary):
    """Encode binary mask as RLE for efficient transfer."""
    import numpy as np
    # Flatten and get runs
    flat = mask_binary.flatten()
    # Pad with zeros at both ends
    padded = np.concatenate([[0], flat, [0]])
    changes = np.where(padded[1:] != padded[:-1])[0]
    runs = changes[1::2] - changes[::2]
    starts = changes[::2]
    return {"starts": starts.tolist(), "lengths": runs.tolist(), "shape": list(mask_binary.shape)}


@app.cls(gpu="A10G", image=sam3_image, secrets=secrets, timeout=300, retries=0)
class Sam3Worker:
    @modal.enter()
    def load_model(self):
        import torch
        import sam3
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # Enable optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Get bpe_path from sam3 package
        sam3_dir = os.path.dirname(sam3.__file__)
        bpe_path = os.path.join(sam3_dir, "assets", "bpe_simple_vocab_16e6.txt.gz")

        print(f"Loading SAM3 model (848M params) with bpe_path: {bpe_path}")

        # Build model - weights should be cached from image build
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
            load_from_HF=True,  # Will use cached weights
        )
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)

        print("SAM3 model loaded successfully")

    @modal.method()
    def segment(
        self,
        image_bytes: bytes,
        points: list[tuple[float, float, str]],
        boxes: list[tuple[float, float, float, float, str]],
        text_prompts: list[str],
    ) -> dict[str, Any]:
        """Run SAM3 segmentation with points, boxes, or text prompts.

        Args:
            image_bytes: Raw image bytes
            points: List of (x, y, polarity) where polarity is "positive" or "negative"
                   Label 1 = positive/foreground, Label 0 = negative/background
            boxes: List of (x1, y1, x2, y2, polarity) in pixel coordinates
            text_prompts: List of text descriptions to segment

        Returns:
            Dict with masks (including RLE-encoded mask data), boxes, scores
        """
        import numpy as np
        import torch
        from PIL import Image

        # Load image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        print(f"Processing image: {width}x{height}")
        print(f"Inputs: {len(points)} points, {len(boxes)} boxes, {len(text_prompts)} text prompts")

        # Set image in processor
        state = self.processor.set_image(img)

        results = {
            "width": width,
            "height": height,
            "masks": [],
            "prompt_points": points,
            "prompt_boxes": boxes,
            "text_prompts": text_prompts,
        }

        # Handle text prompts via processor
        if text_prompts:
            for prompt in text_prompts:
                print(f"Processing text prompt: '{prompt}'")
                text_state = self.processor.set_image(img)
                text_state = self.processor.set_text_prompt(prompt, text_state)

                if "masks" in text_state and len(text_state["masks"]) > 0:
                    for i, mask in enumerate(text_state["masks"]):
                        score = float(text_state["scores"][i]) if "scores" in text_state else 1.0
                        mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
                        mask_data = self._mask_to_data(mask_np, score, width, height)
                        mask_data["prompt_type"] = "text"
                        mask_data["prompt"] = prompt
                        results["masks"].append(mask_data)
                        print(f"  Text mask {i}: score={score:.3f}, area={mask_data['area']}")

        # Handle points and boxes using processor's geometric prompt API
        # SAM3 supports:
        # - Multiple boxes (accumulated in processor state)
        # - Both positive (label=True) and negative (label=False) boxes
        # - Points via predict_inst after boxes are set
        if points or boxes:
            print(f"Processing {len(points)} point(s), {len(boxes)} box(es)")

            # Start with processor state for geometric prompts (boxes)
            prompt_state = state

            # Add all boxes to processor state (supports multiple, positive and negative)
            if boxes:
                print(f"Adding {len(boxes)} box prompt(s) via processor.add_geometric_prompt:")
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2, polarity = box
                    is_positive = polarity == "positive"

                    # Convert pixel coords to normalized cxcywh format
                    center_x = (x1 + x2) / 2.0 / width
                    center_y = (y1 + y2) / 2.0 / height
                    box_w = (x2 - x1) / width
                    box_h = (y2 - y1) / height
                    normalized_box = [center_x, center_y, box_w, box_h]

                    print(f"  Box {i}: ({x1:.1f},{y1:.1f})-({x2:.1f},{y2:.1f}) "
                          f"norm=[{center_x:.3f},{center_y:.3f},{box_w:.3f},{box_h:.3f}] "
                          f"{'positive' if is_positive else 'negative'}")

                    prompt_state = self.processor.add_geometric_prompt(normalized_box, is_positive, prompt_state)

            # Add points via predict_inst (refines the box-based state)
            if points:
                print(f"Adding {len(points)} point prompt(s) via predict_inst:")
                for i, p in enumerate(points):
                    print(f"  Point {i}: ({p[0]:.1f}, {p[1]:.1f}) {'positive' if p[2] == 'positive' else 'negative'}")

                point_coords = np.array([[p[0], p[1]] for p in points])
                point_labels = np.array([1 if p[2] == "positive" else 0 for p in points])

                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    masks, scores, logits = self.model.predict_inst(
                        prompt_state,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                    )

                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                score = float(scores[best_idx])

                print(f"Result: best_idx={best_idx}, score={score:.3f}")

                mask_data = self._mask_to_data(mask, score, width, height)
                mask_data["prompt_type"] = "points" if not boxes else "points+boxes"
                mask_data["prompt"] = {"points": points, "boxes": boxes}
                results["masks"].append(mask_data)

            elif boxes:
                # Box-only: extract masks from processor state
                print("Extracting masks from processor state (box-only)")
                if "masks" in prompt_state and len(prompt_state["masks"]) > 0:
                    for i, mask in enumerate(prompt_state["masks"]):
                        score = float(prompt_state["scores"][i]) if "scores" in prompt_state else 1.0
                        mask_np = mask.cpu().numpy() if hasattr(mask, 'cpu') else np.array(mask)
                        mask_data = self._mask_to_data(mask_np, score, width, height)
                        mask_data["prompt_type"] = "boxes"
                        mask_data["prompt"] = boxes
                        results["masks"].append(mask_data)
                        print(f"  Box mask {i}: score={score:.3f}, area={mask_data['area']}")
                else:
                    print("  No masks in processor state")

        print(f"Returning {len(results['masks'])} masks")
        return results

    def _mask_to_data(self, mask, score: float, width: int, height: int) -> dict:
        """Convert mask array to serializable data with RLE encoding."""
        import numpy as np

        # Ensure 2D numpy array
        if hasattr(mask, 'cpu'):
            mask = mask.cpu().numpy()
        mask = np.array(mask)

        if mask.ndim > 2:
            mask = mask.squeeze()

        # Binarize
        mask_binary = (mask > 0.5).astype(np.uint8)
        area = int(mask_binary.sum())

        # Compute bbox
        if area > 0:
            ys, xs = np.where(mask_binary)
            bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        else:
            bbox = [0, 0, 0, 0]

        # Encode mask as RLE for efficient transfer
        rle = encode_mask_rle(mask_binary)

        return {
            "id": f"mask-{uuid.uuid4().hex[:8]}",
            "score": score,
            "bbox": bbox,
            "area": area,
            "rle": rle,  # RLE-encoded mask for rendering
        }


@app.local_entrypoint()
def main(image_path: str):
    """Test: modal run modal_app.py --image-path path/to/image.png"""
    import json
    data = Path(image_path).read_bytes()
    worker = Sam3Worker()

    # Test with a center point
    result = worker.segment.remote(data, [(320, 240, "positive")], [], [])
    print(json.dumps(result, indent=2))
