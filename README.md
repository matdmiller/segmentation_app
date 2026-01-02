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
- `Dockerfile` – Runtime container for the FastHTML app.
- `docker-compose.yml` – Compose stack with secrets for Modal/HF tokens.
- `docker-entrypoint.sh` – Loads secrets into env vars at container start.
- `tests/Dockerfile.e2e` – Playwright E2E test container.
- `tests/Makefile` – E2E build targets.

## Prerequisites
- Python 3.10+
- `uv` for Python environment + dependency management (`pip install uv` if needed)
- Modal token and Hugging Face token available in the environment (used only during setup; never exposed to the frontend).
- Docker + Docker Compose (for containerized runs/tests)

## One-time setup (local dev)
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

## Running with Docker Compose (recommended for containerized runs)
1. Fill the secrets files (gitignored):
   - `.secrets/modal_token_id`
   - `.secrets/modal_token_secret`
   - `.secrets/huggingface_token`
2. Build and start:
   ```bash
   docker compose up --build
   ```

Open http://localhost:8000 to use the app.

## Running the app container directly (no compose)
If you prefer `docker run`, pass tokens via environment variables:
```bash
docker build -t segmentation-app .
docker run --rm -p 8000:8000 \
  -e MODAL_TOKEN_ID=... \
  -e MODAL_TOKEN_SECRET=... \
  -e HUGGINGFACE_TOKEN=... \
  segmentation-app
```

## Deploying Changes
**IMPORTANT:** After making code changes, always follow this workflow:
```bash
# 1. Rebuild Docker to include latest code changes
docker compose build --no-cache

# 2. Deploy Modal from within the container
docker compose run --rm -e MODAL_TOKEN_ID -e MODAL_TOKEN_SECRET app modal deploy modal_app.py

# 3. Restart the local container to pick up new code
docker compose down && docker compose up -d
```

This ensures both the deployed Modal app AND your local container use the latest code.

Ensure GPU type remains `A10G` when modifying the deployment.
The Modal app name defaults to `sam3-segmentation` and can be overridden with `MODAL_APP_NAME`.
If SAM 3 weights require Hugging Face access, either set `HUGGINGFACE_TOKEN`/`HF_TOKEN` in the deploy environment
or create a Modal secret and set `MODAL_HF_SECRET_NAME` to its name before deploying.

## Downloading Predictions
Use the "Download predictions" button to save the latest predictions JSON, which includes prompts and mask geometry.

## Running Browser Tests (required before finishing changes)
Use the Dockerized E2E runner (builds `tests/Dockerfile.e2e` and runs Playwright in the container, mounting `.secrets` when present):
```bash
make -C tests e2e
```
For a clean rebuild:
```bash
make -C tests e2e-clean
```

Modal-backed segmentation is required by default; ensure `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` are set (or stored in `.secrets/`).
To skip Modal-backed tests (UI-only), set `E2E_SKIP_MODAL=1`.

If you want to run locally instead, install Playwright dependencies and run the tests (the app will be started automatically on a test port):
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
