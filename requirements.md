# Segmentation App Requirements

## Purpose
Build a minimal, elegant FastHTML application that allows users to segment uploaded images using Meta's SAM 3 via Modal with A10G GPUs. The app should keep frontend JavaScript minimal and perform most logic on the backend.

## Functional Requirements
- Users can upload images (photos) for segmentation.
- Support multiple selection modalities for guiding the model:
  - Positive and negative points.
  - Positive and negative bounding boxes.
  - Text prompts (e.g., "select the dog").
- Use Meta's SAM 3 model for segmentation inference.
- Compute must run on Modal using an A10G GPU (no larger GPUs).
- Keep credentials (e.g., Modal token) on the server; never expose tokens to the frontend.
- Allow users to view overlays of predictions and their selections on the original image.
- Display live updates of predictions in a text area while inference runs.
- Provide a way to download all predictions as JSON.
- Include basic browser tests that exercise upload, selection capture, segmentation invocation, and prediction download.
- Keep the UI simple, minimal, and aligned with FastHTML style (locality of behavior).
- Support multiple predictions per session; users can refine selections and re-run segmentation.

## Non-Functional Requirements
- Prefer backend-driven behavior with minimal frontend JavaScript.
- Simple, elegant design with clean defaults; no heavy frontend frameworks.
- Persist temporary uploads server-side only; avoid sending secrets to the client.
- Ensure Modal GPU spec is constrained to `A10G`.
- Keep code organized for maintainability: clear separation between app UI, API endpoints, and Modal compute definitions.
- Provide documentation within the repository for setup, running locally, and deploying with Modal.
- Styling should lean on Monster UI (FrankenUI/Daisy UI theme via FastHTML headers) and vanilla JS without compile steps.
- Always reference the FastHTML docs when adjusting server-rendered UI behavior: https://www.fastht.ml/docs/llms-ctx.txt.
- Keep frontend JavaScript vanilla and minimal—no build tooling or heavy frameworks.
- Use `uv` for Python environment and dependency management; install FastHTML from GitHub, not PyPI:
  `uv pip install -U "git+https://github.com/AnswerDotAI/fasthtml.git" pillow modal monsterui huggingface_hub`.
- Always run the automated tests (including Playwright browser tests) before finishing work.
- Provide a `setup.sh` script that initializes the uv environment, installs JS/browser test dependencies, logs into Modal using
  `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`, and performs a Hugging Face login when `HUGGINGFACE_TOKEN` or `HF_TOKEN` exists in the
  environment (logins occur only during setup execution).
- When compute-related code changes or whenever Modal credentials are available, run the Modal local entrypoint to exercise SAM 3 on A10G with a real image (requires Modal token):
  `modal run modal_app.py --image-path path/to/image.png`.

## Open Questions / Assumptions
- Assumed Modal token and SAM 3 weights are available to the runtime via environment variables and Modal image setup.
- Overlay coloring uses server-side rendering (Pillow) with semi-transparent masks; adjust if a different palette is desired.
- Downloaded JSON includes selections, model metadata, and per-mask geometry.
- Image uploads are stored temporarily in `data/` and cleaned on restart; add retention policy if needed.

Update this document whenever requirements change.
