#!/usr/bin/env bash
set -euo pipefail

# Root of the repo
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

info() { printf "[setup] %s\n" "$*"; }
warn() { printf "[setup][warn] %s\n" "$*" >&2; }

# Ensure uv is available
if ! command -v uv >/dev/null 2>&1; then
  warn "uv is not installed. Install via 'pip install uv' before rerunning setup."
  exit 1
fi

info "Creating virtual environment with uv..."
uv venv
# shellcheck disable=SC1091
source .venv/bin/activate

info "Installing Python dependencies with uv pip..."
uv pip install -U "git+https://github.com/AnswerDotAI/fasthtml.git" pillow modal monsterui huggingface_hub

info "Ensuring temp directories exist..."
mkdir -p data tests/tmp

# Modal authentication (requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET)
modal_profile="${MODAL_PROFILE:-default}"
if [[ -n "${MODAL_TOKEN_ID:-}" && -n "${MODAL_TOKEN_SECRET:-}" ]]; then
  info "Configuring Modal token for profile '${modal_profile}'..."
  modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET" --profile "$modal_profile" --force
else
  warn "Modal token env vars (MODAL_TOKEN_ID and MODAL_TOKEN_SECRET) not fully set; skipping Modal login."
fi

# Hugging Face authentication (HUGGINGFACE_TOKEN or HF_TOKEN)
hf_token="${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}"
if [[ -n "$hf_token" ]]; then
  info "Logging into Hugging Face from setup.sh..."
  huggingface-cli login --token "$hf_token" --add-to-git-credential
else
  warn "Hugging Face token not provided; skipping Hugging Face login."
fi

# Node/Playwright setup for browser tests
if command -v npm >/dev/null 2>&1; then
  info "Installing npm dependencies..."
  npm install
  info "Installing Playwright chromium browser..."
  npx playwright install chromium
  info "Installing Playwright system dependencies for chromium..."
  npx playwright install-deps chromium || warn "Playwright system deps install encountered an issue; rerun if browsers fail"
else
  warn "npm not found; skipping JS dependency installation."
fi

info "Setup complete. Activate the env with 'source .venv/bin/activate' before running the app."
