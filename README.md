# SAM 3 FastHTML Segmentation App

This repository hosts a minimal FastHTML web app that runs Meta's SAM 3 image segmentation on Modal with A10G GPUs.

## Features
- Upload an image and guide segmentation with points, bounding boxes, or text prompts (positive/negative).
- Runs SAM 3 inference on Modal using an A10G GPU and keeps tokens server-side.
- Live-updated predictions text area and downloadable JSON of all predictions.
- Overlays selections and predicted masks on top of the uploaded image.

## Project Layout
- `app.py` – FastHTML frontend and API endpoints.
- `modal_app.py` – Modal deployment for SAM 3 inference on an A10G GPU.
- `requirements.md` – Current requirements and open assumptions.
- `data/` – Temporary image uploads (not committed).

## Prerequisites
- Python 3.10+
- `uv` for Python environment + dependency management (`pip install uv` if needed)
- Modal token and Hugging Face token available in the environment (used only during setup; never exposed to the frontend).

## One-time setup
Run the provided setup script to provision the Python/Node environments, configure Modal, and (optionally) log in to Hugging Face
if a token exists in the environment:
```bash
./setup.sh
```
The script:
- Creates a uv-managed virtual environment and installs dependencies.
- Configures Modal auth via `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` (profile defaults to `${MODAL_PROFILE:-default}`).
- Performs a Hugging Face CLI login only when `HUGGINGFACE_TOKEN` or `HF_TOKEN` is set.
- Installs npm dependencies, Playwright's Chromium, and required system dependencies for browser tests.

## Running Locally (uv)
1. Activate the environment after running setup:
   ```bash
   source .venv/bin/activate
   ```
2. Start the UI (FastHTML + Monster UI headers):
   ```bash
   python app.py
   ```
3. In another terminal, run Modal locally or deploy if you want live SAM 3 inference:
   ```bash
   modal run modal_app.py
   ```

Open http://localhost:8000 to use the app.

## Deploying to Modal
Deploy the SAM 3 worker:
```bash
modal deploy modal_app.py
```

Ensure GPU type remains `A10G` when modifying the deployment.

## Downloading Predictions
Use the "Download predictions" button to save the latest predictions JSON, which includes prompts and mask geometry.

## Running Browser Tests (required before finishing changes)
Install Playwright dependencies and run the tests (the app will be started automatically on a test port):
```bash
npm install
npx playwright install chromium
npm test
```
Always run these tests before completing work or preparing a PR.

## Modal smoke test (required when modifying compute or when Modal credentials are available)
With Modal credentials configured (available via environment variables in this repo), verify the SAM 3 worker end-to-end on A10G by running:
```bash
modal run modal_app.py --image-path path/to/local_image.png
```
Always run this smoke test when touching compute-related code or when credentials are present so the Modal function is exercised against real hardware; ensure the GPU spec remains `A10G`.

## UI Notes
- Styling uses Monster UI (FrankenUI/Daisy) via `Theme.blue.headers(mode="dark", daisy=True)`.
- Keep frontend JS vanilla (no build tooling) and lean on FastHTML locality of behavior.
- Reference the FastHTML docs when adjusting UI structure: https://www.fastht.ml/docs/llms-ctx.txt


## Updating Requirements
Keep `requirements.md` in sync with any new or changed requirements.
