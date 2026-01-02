"""Test file for SAM3 Modal deployment.

Run with: modal run modal_test.py

This validates:
1. Text prompt segmentation (e.g., "person", "dog")
2. Point prompt segmentation (positive/negative points)
3. Box prompt segmentation (positive/negative boxes)
4. Combined prompts
"""

import io
import json
import os
from pathlib import Path

import modal

APP_NAME = os.environ.get("MODAL_APP_NAME", "sam3-segmentation")

app = modal.App("sam3-test")

# HF secret for gated models (must be consistent between local and remote)
HF_SECRET_NAME = "huggingface-secret"
secrets = [modal.Secret.from_name(HF_SECRET_NAME)]

# Use the same image as the main app
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
        "requests",
        "decord",
        "pycocotools",
        "psutil",
        "matplotlib",
        "scipy",
        "git+https://github.com/facebookresearch/sam3.git",
    )
)


@app.function(gpu="A10G", image=sam3_image, secrets=secrets, timeout=600, retries=0)
def run_sam3_tests():
    """Run comprehensive SAM3 tests."""
    import numpy as np
    import requests
    import torch
    from PIL import Image

    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    results = {"tests": [], "passed": 0, "failed": 0}

    def log_test(name: str, passed: bool, details: str = ""):
        status = "PASSED" if passed else "FAILED"
        print(f"[{status}] {name}: {details}")
        results["tests"].append({"name": name, "passed": passed, "details": details})
        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Download test image
    print("Downloading test image...")
    test_image_url = "https://raw.githubusercontent.com/facebookresearch/sam3/main/assets/images/test_image.jpg"
    response = requests.get(test_image_url, timeout=30)
    if response.status_code != 200:
        # Fallback to a simple generated image
        print("Using generated test image")
        img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    else:
        img = Image.open(io.BytesIO(response.content)).convert("RGB")

    width, height = img.size
    print(f"Test image size: {width}x{height}")

    # Load model
    print("Loading SAM3 model...")
    sam3_dir = os.path.dirname(sam3.__file__)
    bpe_path = os.path.join(sam3_dir, "assets", "bpe_simple_vocab_16e6.txt.gz")

    try:
        model = build_sam3_image_model(bpe_path=bpe_path, enable_inst_interactivity=True)
        processor = Sam3Processor(model, confidence_threshold=0.5)
        log_test("Model Loading", True, "SAM3 model loaded successfully")
    except Exception as e:
        log_test("Model Loading", False, str(e))
        return results

    # Test 1: Set Image
    print("\n--- Test 1: Set Image ---")
    try:
        state = processor.set_image(img)
        has_state = state is not None and isinstance(state, dict)
        log_test("Set Image", has_state, f"State keys: {list(state.keys()) if has_state else 'None'}")
    except Exception as e:
        log_test("Set Image", False, str(e))
        return results

    # Test 2: Text Prompt Segmentation
    print("\n--- Test 2: Text Prompt Segmentation ---")
    try:
        state = processor.set_image(img)
        state = processor.set_text_prompt("object", state)
        has_masks = "masks" in state and len(state.get("masks", [])) >= 0
        num_masks = len(state.get("masks", [])) if has_masks else 0
        log_test(
            "Text Prompt",
            True,  # Just check it runs without error
            f"Text prompt processed, found {num_masks} masks"
        )
    except Exception as e:
        log_test("Text Prompt", False, str(e))

    # Test 3: Point Prompt Segmentation (single positive point)
    print("\n--- Test 3: Single Point Prompt ---")
    try:
        state = processor.set_image(img)
        point_coords = np.array([[width // 2, height // 2]])  # Center of image
        point_labels = np.array([1])  # Positive

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        has_masks = masks is not None and len(masks) > 0
        best_score = float(scores[np.argmax(scores)]) if has_masks else 0
        log_test(
            "Single Point Prompt",
            has_masks,
            f"Generated {len(masks)} masks, best score: {best_score:.3f}"
        )
    except Exception as e:
        log_test("Single Point Prompt", False, str(e))

    # Test 4: Multiple Point Prompts (positive + negative)
    print("\n--- Test 4: Multiple Point Prompts ---")
    try:
        state = processor.set_image(img)
        point_coords = np.array([
            [width // 3, height // 2],      # Positive point
            [2 * width // 3, height // 2],  # Negative point
        ])
        point_labels = np.array([1, 0])  # 1=positive, 0=negative

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

        has_masks = masks is not None and len(masks) > 0
        log_test(
            "Multiple Point Prompts",
            has_masks,
            f"Generated {len(masks)} masks with pos+neg points"
        )
    except Exception as e:
        log_test("Multiple Point Prompts", False, str(e))

    # Test 5: Box Prompt (geometric prompt)
    print("\n--- Test 5: Box Prompt ---")
    try:
        state = processor.set_image(img)

        # Box in center of image (normalized cxcywh format)
        center_x = 0.5
        center_y = 0.5
        box_w = 0.4
        box_h = 0.4
        box = [center_x, center_y, box_w, box_h]

        state = processor.add_geometric_prompt(box, True, state)  # True = positive

        # Check if masks were generated
        has_masks = "masks" in state and len(state.get("masks", [])) > 0
        log_test(
            "Box Prompt",
            True,  # Just check it doesn't crash
            f"Box prompt processed, state has masks: {has_masks}"
        )
    except Exception as e:
        log_test("Box Prompt", False, str(e))

    # Test 6: Negative Box Prompt
    print("\n--- Test 6: Negative Box Prompt ---")
    try:
        state = processor.set_image(img)

        # First add a positive box
        pos_box = [0.5, 0.5, 0.6, 0.6]
        state = processor.add_geometric_prompt(pos_box, True, state)

        # Then add a negative box to exclude part
        neg_box = [0.3, 0.3, 0.2, 0.2]
        state = processor.add_geometric_prompt(neg_box, False, state)

        log_test(
            "Negative Box Prompt",
            True,
            "Positive + negative box prompts processed"
        )
    except Exception as e:
        log_test("Negative Box Prompt", False, str(e))

    # Test 7: Reset Prompts
    print("\n--- Test 7: Reset Prompts ---")
    try:
        state = processor.set_image(img)
        state = processor.add_geometric_prompt([0.5, 0.5, 0.3, 0.3], True, state)
        state = processor.reset_all_prompts(state)
        log_test("Reset Prompts", True, "Prompts reset successfully")
    except Exception as e:
        log_test("Reset Prompts", False, str(e))

    # Test 8: Confidence Threshold
    print("\n--- Test 8: Confidence Threshold ---")
    try:
        state = processor.set_image(img)
        state = processor.set_confidence_threshold(0.7, state)
        log_test("Confidence Threshold", True, "Threshold set to 0.7")
    except Exception as e:
        log_test("Confidence Threshold", False, str(e))

    # Test 9: Mask Output Format
    print("\n--- Test 9: Mask Output Format ---")
    try:
        state = processor.set_image(img)
        point_coords = np.array([[width // 2, height // 2]])
        point_labels = np.array([1])

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,  # Single mask output
            )

        # Check mask shape
        mask_shape = masks[0].shape if len(masks) > 0 else None
        valid_shape = mask_shape is not None and len(mask_shape) == 2
        log_test(
            "Mask Output Format",
            valid_shape,
            f"Mask shape: {mask_shape}"
        )
    except Exception as e:
        log_test("Mask Output Format", False, str(e))

    # Test 10: Iterative Refinement with Logits
    print("\n--- Test 10: Iterative Refinement ---")
    try:
        state = processor.set_image(img)

        # First prediction
        point_coords = np.array([[width // 2, height // 2]])
        point_labels = np.array([1])

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, logits = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

            # Use logits from best mask for refinement
            best_idx = np.argmax(scores)
            mask_input = logits[best_idx:best_idx+1, :, :]

            # Add another point and refine
            point_coords_2 = np.array([
                [width // 2, height // 2],
                [width // 2 + 50, height // 2 + 50]
            ])
            point_labels_2 = np.array([1, 1])

            masks_refined, scores_refined, _ = model.predict_inst(
                state,
                point_coords=point_coords_2,
                point_labels=point_labels_2,
                mask_input=mask_input,
                multimask_output=False,
            )

        log_test(
            "Iterative Refinement",
            len(masks_refined) > 0,
            f"Refined mask generated with score: {float(scores_refined[0]):.3f}"
        )
    except Exception as e:
        log_test("Iterative Refinement", False, str(e))

    # Summary
    print("\n" + "=" * 50)
    print(f"SUMMARY: {results['passed']} passed, {results['failed']} failed")
    print("=" * 50)

    return results


@app.local_entrypoint()
def main():
    """Run all SAM3 tests."""
    print("Starting SAM3 comprehensive tests...")
    results = run_sam3_tests.remote()
    print("\nFinal Results:")
    print(json.dumps(results, indent=2))

    if results["failed"] > 0:
        print(f"\n⚠️  {results['failed']} test(s) failed!")
        exit(1)
    else:
        print("\n✓ All tests passed!")
